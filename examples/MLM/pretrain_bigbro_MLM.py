import pytorch_lightning as pl
import sys
import transformers
import torch

import bigbro
import iterable


class TrainHarness(pl.LightningModule):
   "A Lightning train harness with AdamW and CLR."
   def __init__(self, model):
      super().__init__()
      self.model = model

   def configure_optimizers(self):
      # Use lr 5e-5, AdamW and linear decay with warmup.
      opt = torch.optim.AdamW(self.parameters(), lr=5e-5)
      # Other possible variants to explore when running on
      # different hardware.
      #import deepspeed
      #opt = deepspeed.ops.adam.DeepSpeedCPUAdam(self.parameters(), lr=5e-5)
      #opt = deepspeed.ops.adam.FusedAdam(self.parameters(), lr=5e-5)
      scheduler = torch.optim.lr_scheduler.CyclicLR(
           optimizer = opt,
           base_lr = 1e-5,
           max_lr = 5e-5,
           cycle_momentum = False
      )
      return [opt], [{"scheduler": scheduler, "interval": "step"}]

   def training_step(self, batch, batch_idx):
      outputs = self.model(**batch)
      return outputs.loss


class DataForMLM(pl.LightningDataModule):
   "A lightning data module for MLM on JSON data."
   def __init__(
        self,
        train_data_path,
        tokenizer,
        batch_size = 1
   ):
      super().__init__()
      self.train_data_path = train_data_path
      self.tokenizer = tokenizer
      self.batch_size = batch_size
      self.collator = transformers.DataCollatorForLanguageModeling(tokenizer)

   def collate(self, examples):
      tokenized = tokenizer(
           [ex["description"] for ex in examples],
           return_token_type_ids = False,
           return_attention_mask = False,
           truncation = True,
           max_length = 16384
      )
      return self.collator(tokenized["input_ids"])

   def train_dataloader(self):
      return torch.utils.data.DataLoader(
           dataset = iterable.IterableJSONData(self.train_data_path),
           collate_fn = self.collate,
           batch_size = self.batch_size,
           num_workers = 2,
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
       max_position_embeddings = 16384, sep_token_id = 2,
       # Config options for the RoFormer.
       embedding_size = 768, rotary_value = False)
   model = bigbro.BigBroForMaskedLM(config=config)
   harnessed_model = TrainHarness(model)


   GPU_free, GPU_total = torch.cuda.mem_get_info()
   s = GPU_total / 51047628800 # Scaling factor.
   
   data_M = DataForMLM(train_data_path_M, tokenizer, batch_size=int(s*4))
   data_L = DataForMLM(train_data_path_L, tokenizer, batch_size=int(s*2))
   data_XL = DataForMLM(train_data_path_XL, tokenizer, batch_size=int(s*1))

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
   trainer.limit_train_batches = 11
   trainer.fit(harnessed_model, data_M)

   trainer.fit_loop.max_epochs = 2
   trainer.limit_train_batches = 7
   trainer.fit(harnessed_model, data_L)

   trainer.fit_loop.max_epochs = 3
   trainer.limit_train_batches = 2
   trainer.fit(harnessed_model, data_XL)

   trainer.save_checkpoint("./checkpoints")
   model.bert.save_to_file(trained_model_path)
