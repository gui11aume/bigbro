import pytorch_lightning as pl
import sys
import transformers
import torch

import bigbro
import iterable


class TrainHarness(pl.LightningModule):
    "A Lightning train harness with AdamW and CLR."
    def __init__(self, model, base_lr=5e-5):
        super().__init__()
        self.model = model
        self.base_lr = base_lr

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.base_lr)
        #import deepspeed
        #opt = deepspeed.ops.adam.DeepSpeedCPUAdam(self.parameters(), lr=5e-5)
        #opt = deepspeed.ops.adam.FusedAdam(self.parameters(), lr=5e-5)
        return opt

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu = False,
        using_native_amp = False,
        using_lbfgs = False,
    ):
        # Using 'torch.optim.lr_scheduler.CyclicLR' in 'configure_optimizer'
        # gives an error "cannot pickle 'WeakMethod' object" upon saving
        # a training checkpoint. This is a re-implementation to go around
        # the bug (pytorch 1.13.0 pytorch_lightning 1.7.7).
        optimizer.step(closure=optimizer_closure)
        # Cyclic learning rate.
        phase = self.trainer.global_step % 4000
        lr_scale = .2 + .8 * (1. - abs(phase / 2000 - 1.))
        for pg in optimizer.param_groups:
            pg["lr"] = lr_scale * self.base_lr

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        return outputs.loss


class DataForMLM(pl.LightningDataModule):
    "A lightning data module for MLM on JSON data."
    def __init__(
          self,
          train_data_path,
          tokenizer,
          max_length = 16384,
          batch_size = 1
    ):
        super().__init__()
        self.train_data_path = train_data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.collator = transformers.DataCollatorForLanguageModeling(tokenizer)

    def collate(self, examples):
        tokenized = self.tokenizer(
              [ex["description"] for ex in examples],
              return_token_type_ids = False,
              return_attention_mask = False,
              truncation = True,
              max_length = self.max_length,
        )
        return self.collator(tokenized["input_ids"])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
              dataset = iterable.IterableJSONData(self.train_data_path),
              collate_fn = self.collate,
              batch_size = self.batch_size,
              num_workers = 4,
              persistent_workers = True
        )


if __name__ == "__main__":

   tokenizer_path = sys.argv[1]
   train_data_path_M = sys.argv[2]
   train_data_path_L = sys.argv[3]
   train_data_path_XL = sys.argv[4]
   trained_model_path = sys.argv[5]

   tokenizer = transformers.PreTrainedTokenizerFast(
         tokenizer_file = tokenizer_path,
         bos_token = "[CLS]",
         eos_token = "[SEP]",
         unk_token = "[UNK]",
         sep_token = "[SEP]",
         pad_token = "[PAD]",
         cls_token = "[CLS]",
         mask_token = "[MASK]"
   )

   # The mighty BigBro model (BigBird with RoFormer position encoding).
   config = transformers.BigBirdConfig(vocab_size=len(tokenizer),
         #attention_type = "original_full", 
         attention_type = "block_sparse",
         max_position_embeddings = 24576, sep_token_id = 2,
         # Config options for the RoFormer.
         embedding_size = 768, rotary_value = False)
   model = bigbro.BigBroForMaskedLM(config=config)
   harnessed_model = TrainHarness(model)

   # The documents should not exceed specified length in each bucket,
   # but we still truncate as a safety mechanism.
   data_M = DataForMLM(train_data_path_M, tokenizer, batch_size=4, max_length=8192)
   data_L = DataForMLM(train_data_path_L, tokenizer, batch_size=2, max_length=16384)
   data_XL = DataForMLM(train_data_path_XL, tokenizer, batch_size=1, max_length=24576)

   trainer = pl.Trainer(
         default_root_dir = "./checkpoints",
         strategy = pl.strategies.DeepSpeedStrategy(
             # No offloading, use AdamW (fastest strategy).
             stage = 2,
             offload_optimizer = False,
             offload_parameters = False,
         ),
         accelerator = "gpu",
         devices = torch.cuda.device_count(),
         precision = 16,
   )

   # Inspired from https://github.com/Lightning-AI/lightning/issues/8435
   trainer.fit_loop.max_epochs = 1
   trainer.limit_train_batches = 30000
   trainer.fit(harnessed_model, data_M)

   trainer.fit_loop.max_epochs = 2
   trainer.limit_train_batches = 15000
   trainer.fit(harnessed_model, data_L)

   trainer.fit_loop.max_epochs = 3
   trainer.limit_train_batches = 7500
   trainer.fit(harnessed_model, data_XL)

   trainer.save_checkpoint("./pretrained_majestic_BigBro")
   model.bert.save_to_file(trained_model_path)
