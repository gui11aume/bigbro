import deepspeed
import transformers
import torch

logger = transformers.utils.logging.get_logger(__name__)

# BigBro is an extension of BigBird with RoFormer position encoding.
class BigBroBlockSparseAttention(transformers.models.big_bird.modeling_big_bird.BigBirdBlockSparseAttention):
    def __init__(self, config, seed=None):
        # =================
        #super().__init__()
        super(transformers.models.big_bird.modeling_big_bird.BigBirdBlockSparseAttention, self).__init__()
        # =================

        self.max_seqlen = config.max_position_embeddings
        self.seed = seed

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size {config.hidden_size} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.num_random_blocks = config.num_random_blocks
        self.block_size = config.block_size

        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = torch.nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)
        self.key = torch.nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)
        self.value = torch.nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)
        # =================
        self.rotary_value = config.rotary_value
        # =================

    # =================
    #def forward(
    #    self,
    #    hidden_states,
    #    band_mask=None,
    #    from_mask=None,
    #    to_mask=None,
    #    from_blocked_mask=None,
    #    to_blocked_mask=None,
    #    output_attentions=None,
    #):
    def forward(
        self,
        hidden_states,
        sinusoidal_pos=None, # <===
        band_mask=None,
        from_mask=None,
        to_mask=None,
        from_blocked_mask=None,
        to_blocked_mask=None,
        output_attentions=None,
    ):
    # =================
        # Currently this `class` can't be used in decoder.

        batch_size, seqlen, _ = hidden_states.size()
        to_seq_length = from_seq_length = seqlen
        from_block_size = to_block_size = self.block_size

        assert from_seq_length % from_block_size == 0, "Query sided sequence length must be multiple of block size"
        assert to_seq_length % to_block_size == 0, "Key/Value sided sequence length must be multiple of block size"

        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        # =================
        if sinusoidal_pos is not None:
            if self.rotary_value:
                query_layer, key_layer, value_layer = self.apply_rotary_position_embeddings(
                    sinusoidal_pos, query_layer, key_layer, value_layer
                )
            else:
                query_layer, key_layer = self.apply_rotary_position_embeddings(
                    sinusoidal_pos, query_layer, key_layer
                )
        # =================

        context_layer, attention_probs = self.bigbird_block_sparse_attention(
            query_layer,
            key_layer,
            value_layer,
            band_mask,
            from_mask,
            to_mask,
            from_blocked_mask,
            to_blocked_mask,
            self.num_attention_heads,
            self.num_random_blocks,
            self.attention_head_size,
            from_block_size,
            to_block_size,
            batch_size,
            from_seq_length,
            to_seq_length,
            seed=self.seed,
            plan_from_length=None,
            plan_num_rand_blocks=None,
            output_attentions=output_attentions,
        )

        context_layer = context_layer.contiguous().view(batch_size, from_seq_length, -1)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

    @staticmethod
    def apply_rotary_position_embeddings(sinusoidal_pos, query_layer, key_layer, value_layer=None):
        # https://kexue.fm/archives/8265
        # sin [batch_size, num_heads, sequence_length, embed_size_per_head//2]
        # cos [batch_size, num_heads, sequence_length, embed_size_per_head//2]
        sin, cos = sinusoidal_pos.chunk(2, dim=-1)
        # sin [θ0,θ1,θ2......θd/2-1] -> sin_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        sin_pos = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos)
        # cos [θ0,θ1,θ2......θd/2-1] -> cos_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        cos_pos = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)
        # rotate_half_query_layer [-q1,q0,-q3,q2......,-qd-1,qd-2]
        rotate_half_query_layer = torch.stack([-query_layer[..., 1::2], query_layer[..., ::2]], dim=-1).reshape_as(
            query_layer
        )
        query_layer = query_layer * cos_pos + rotate_half_query_layer * sin_pos
        # rotate_half_key_layer [-k1,k0,-k3,k2......,-kd-1,kd-2]
        rotate_half_key_layer = torch.stack([-key_layer[..., 1::2], key_layer[..., ::2]], dim=-1).reshape_as(key_layer)
        key_layer = key_layer * cos_pos + rotate_half_key_layer * sin_pos
        if value_layer is not None:
            # rotate_half_value_layer [-v1,v0,-v3,v2......,-vd-1,vd-2]
            rotate_half_value_layer = torch.stack([-value_layer[..., 1::2], value_layer[..., ::2]], dim=-1).reshape_as(
                value_layer
            )
            value_layer = value_layer * cos_pos + rotate_half_value_layer * sin_pos
            return query_layer, key_layer, value_layer
        return query_layer, key_layer


class BigBroAttention(transformers.models.big_bird.modeling_big_bird.BigBirdAttention):
    def __init__(self, config, seed=None):
        # =================
        #super().__init__()
        super(transformers.models.big_bird.modeling_big_bird.BigBirdAttention, self).__init__()
        # =================
        self.attention_type = config.attention_type
        self.config = config
        self.seed = seed

        # =================
        #if self.config.attention_type == "original_full":
        #    self.self = BigBirdSelfAttention(config)
        #elif self.config.attention_type == "block_sparse":
        #    self.self = BigBirdBlockSparseAttention(config, seed)
        if self.config.attention_type == "original_full":
            self.self = transformers.models.roformer.modeling_roformer.RoFormerSelfAttention(config)
        elif self.config.attention_type == "block_sparse":
            self.self = BigBroBlockSparseAttention(config, seed)
        # =================
        else:
            raise ValueError(
                f"attention_type can either be original_full or block_sparse, but is {self.config.attention_type}"
            )

        # =================
        #self.output = BigBirdSelfOutput(config)
        self.output = transformers.models.big_bird.modeling_big_bird.BigBirdSelfOutput(config)
        # =================

    def set_attention_type(self, value: str):
        if value not in ["original_full", "block_sparse"]:
            raise ValueError(
                f"attention_type can only be set to either 'original_full' or 'block_sparse', but is {value}"
            )
        # attention type is already correctly set
        if value == self.attention_type:
            return

        self.attention_type = value
        if value == "original_full":
            # copy all weights to new full attention class
            # =================
            #attn_weights = BigBirdSelfAttention(self.config)
            attn_weights = transformers.models.roformer.modeling_roformer.RoFormerSelfAttention(self.config)
            # =================
        else:
            # copy all weights to new sparse attention class
            # =================
            #attn_weights = BigBirdBlockSparseAttention(self.config, self.seed)
            attn_weights = BigBroBlockSparseAttention(self.config, self.seed)
            # =================

        attn_weights.query = self.self.query
        attn_weights.value = self.self.value
        attn_weights.key = self.self.key
        self.self = attn_weights
        self.attention_type = value

        if not self.training:
            self.self.eval()

    # =================
    #def forward(
    #    self,
    #    hidden_states,
    #    attention_mask=None,
    #    head_mask=None,
    #    encoder_hidden_states=None,
    #    encoder_attention_mask=None,
    #    past_key_value=None,
    #    output_attentions=False,
    #    # block_sparse config
    #    band_mask=None,
    #    from_mask=None,
    #    to_mask=None,
    #    from_blocked_mask=None,
    #    to_blocked_mask=None,
    #):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        sinusoidal_pos=None, # <===
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        # block_sparse config
        band_mask=None,
        from_mask=None,
        to_mask=None,
        from_blocked_mask=None,
        to_blocked_mask=None,
    ):
    # =================

        # fp16 compatibility
        if band_mask is not None:
            band_mask = band_mask.to(hidden_states.dtype)
        if from_mask is not None:
            from_mask = from_mask.to(hidden_states.dtype)
        if to_mask is not None:
            to_mask = to_mask.to(hidden_states.dtype)

        if self.attention_type == "original_full":
            # =================
            #self_outputs = self.self(
            #    hidden_states,
            #    attention_mask,
            #    head_mask,
            #    encoder_hidden_states,
            #    encoder_attention_mask,
            #    past_key_value,
            #    output_attentions,
            #)
            self_outputs = self.self(
                hidden_states,
                attention_mask,
                sinusoidal_pos, # <===
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )
            # =================
        else:
            assert (
                encoder_hidden_states is None
            ), "BigBird cannot be used as a decoder when config.attention_type != 'original_full'"
            # =================
            #self_outputs = self.self(
            #    hidden_states, band_mask, from_mask, to_mask, from_blocked_mask, to_blocked_mask, output_attentions
            #)
            # Use DeepSpeed checkpointing on attention layers.
            self_outputs = deepspeed.checkpointing.checkpoint(self.self,
                hidden_states, sinusoidal_pos, band_mask, from_mask, to_mask, from_blocked_mask, to_blocked_mask, output_attentions
            )
            # =================

        attention_output = self.output(self_outputs[0], hidden_states)
        # =================
        # 'self_outputs' is a tensor and not a tuple.
        #outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        outputs = (attention_output,) + (self_outputs[1:],)  # add attentions if we output them
        # =================
        return outputs


class BigBroLayer(transformers.models.big_bird.modeling_big_bird.BigBirdLayer):
    def __init__(self, config, seed=None):
        # =================
        #super().__init__()
        super(transformers.models.big_bird.modeling_big_bird.BigBirdLayer, self).__init__()
        # =================
        self.config = config
        self.attention_type = config.attention_type
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        # =================
        #self.attention = BigBirdAttention(config, seed=seed)
        self.attention = BigBroAttention(config, seed=seed)
        # =================
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        # =================
        #if self.add_cross_attention:
        #    assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
        #    self.crossattention = BigBirdAttention(config)
        #self.intermediate = BigBirdIntermediate(config)
        #self.output = BigBirdOutput(config)
        if self.add_cross_attention:
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = BigBroAttention(config)
        self.intermediate = transformers.models.big_bird.modeling_big_bird.BigBirdIntermediate(config)
        self.output = transformers.models.big_bird.modeling_big_bird.BigBirdOutput(config)
        # =================

    # =================
    #def forward(
    #    self,
    #    hidden_states,
    #    attention_mask=None,
    #    head_mask=None,
    #    encoder_hidden_states=None,
    #    encoder_attention_mask=None,
    #    band_mask=None,
    #    from_mask=None,
    #    to_mask=None,
    #    blocked_encoder_mask=None,
    #    past_key_value=None,
    #    output_attentions=False,
    #):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        sinusoidal_pos=None, # <===
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        band_mask=None,
        from_mask=None,
        to_mask=None,
        blocked_encoder_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
    # =================
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # =================
        #self_attention_outputs = self.attention(
        #    hidden_states,
        #    attention_mask,
        #    head_mask,
        #    encoder_hidden_states=encoder_hidden_states,
        #    encoder_attention_mask=encoder_attention_mask,
        #    past_key_value=self_attn_past_key_value,
        #    output_attentions=output_attentions,
        #    band_mask=band_mask,
        #    from_mask=from_mask,
        #    to_mask=to_mask,
        #    from_blocked_mask=blocked_encoder_mask,
        #    to_blocked_mask=blocked_encoder_mask,
        #)
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            sinusoidal_pos, # <===
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_value=self_attn_past_key_value,
            output_attentions=output_attentions,
            band_mask=band_mask,
            from_mask=from_mask,
            to_mask=to_mask,
            from_blocked_mask=blocked_encoder_mask,
            to_blocked_mask=blocked_encoder_mask,
        )
        # =================
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with \
                    cross-attention layers by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # =================
            #cross_attention_outputs = self.crossattention(
            #    attention_output,
            #    attention_mask,
            #    head_mask,
            #    encoder_hidden_states,
            #    encoder_attention_mask,
            #    cross_attn_past_key_value,
            #    output_attentions,
            #)
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                sinusoidal_pos, # <===
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            # =================
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = transformers.modeling_utils.apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )

        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs


class BigBroEncoder(transformers.models.big_bird.modeling_big_bird.BigBirdEncoder):
    def __init__(self, config):
        # =================
        #super().__init__()
        super(transformers.models.big_bird.modeling_big_bird.BigBirdEncoder, self).__init__()
        # =================
        self.config = config
        self.attention_type = config.attention_type
        # =================
        #self.layer = nn.ModuleList(
        #    [BigBirdLayer(config, seed=layer_idx) for layer_idx in range(config.num_hidden_layers)]
        #)
        self.embed_positions = transformers.models.roformer.modeling_roformer.RoFormerSinusoidalPositionalEmbedding(
            config.max_position_embeddings, config.hidden_size // config.num_attention_heads
        )
        self.layer = torch.nn.ModuleList(
            [BigBroLayer(config, seed=layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # =================
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        band_mask=None,
        from_mask=None,
        to_mask=None,
        blocked_encoder_mask=None,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # =================
        sinusoidal_pos = self.embed_positions(hidden_states.shape[:-1])[None, None, :, :]
        # =================

        next_decoder_cache = () if use_cache else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                # =================
                #layer_outputs = torch.utils.checkpoint.checkpoint(
                #    create_custom_forward(layer_module),
                #    hidden_states,
                #    attention_mask,
                #    layer_head_mask,
                #    encoder_hidden_states,
                #    encoder_attention_mask,
                #    band_mask,
                #    from_mask,
                #    to_mask,
                #    blocked_encoder_mask,
                #)
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    sinusoidal_pos, # <===
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    band_mask,
                    from_mask,
                    to_mask,
                    blocked_encoder_mask,
                )
                # =================
            else:

                # =================
                #layer_outputs = layer_module(
                #    hidden_states,
                #    attention_mask,
                #    layer_head_mask,
                #    encoder_hidden_states,
                #    encoder_attention_mask,
                #    band_mask,
                #    from_mask,
                #    to_mask,
                #    blocked_encoder_mask,
                #    past_key_value,
                #    output_attentions,
                #)
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    sinusoidal_pos, # <===
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    band_mask,
                    from_mask,
                    to_mask,
                    blocked_encoder_mask,
                    past_key_value,
                    output_attentions,
                )
                # =================

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class BigBroModel(transformers.BigBirdModel):
    def __init__(self, config, add_pooling_layer=True):
        # =================
        #super().__init__(config)
        super(transformers.BigBirdModel, self).__init__(config)
        # =================
        self.attention_type = self.config.attention_type
        self.config = config

        self.block_size = self.config.block_size

        # =================
        #self.embeddings = BigBirdEmbeddings(config)
        #self.encoder = BigBirdEncoder(config)
        self.embeddings = transformers.models.roformer.modeling_roformer.RoFormerEmbeddings(config)
        if config.embedding_size != config.hidden_size:
            self.embeddings_project = torch.nn.Linear(config.embedding_size, config.hidden_size)
        self.encoder = BigBroEncoder(config)
        # =================

        if add_pooling_layer:
            self.pooler = torch.nn.Linear(config.hidden_size, config.hidden_size)
            self.activation = torch.nn.Tanh()
        else:
            self.pooler = None
            self.activation = None

        if self.attention_type != "original_full" and config.add_cross_attention:
            logger.warning(
                "When using `BigBirdForCausalLM` as decoder, then `attention_type` must be `original_full`. Setting `attention_type=original_full`"
            )
            self.set_attention_type("original_full")

        self.init_weights()
    
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, torch.nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        # =================
        elif isinstance(module, transformers.models.roformer.modeling_roformer.RoFormerSinusoidalPositionalEmbedding):
            pass
        # =================
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # in order to use block_sparse attention, sequence_length has to be at least
        # bigger than all global attentions: 2 * block_size
        # + sliding tokens: 3 * block_size
        # + random tokens: 2 * num_random_blocks * block_size
        max_tokens_to_attend = (5 + 2 * self.config.num_random_blocks) * self.config.block_size
        if self.attention_type == "block_sparse" and seq_length <= max_tokens_to_attend:
            # change attention_type from block_sparse to original_full
            sequence_length = input_ids.size(1) if input_ids is not None else inputs_embeds.size(1)
            logger.warning(
                "Attention type 'block_sparse' is not possible if sequence_length: "
                f"{sequence_length} <= num global tokens: 2 * config.block_size "
                "+ min. num sliding tokens: 3 * config.block_size "
                "+ config.num_random_blocks * config.block_size "
                "+ additional buffer: config.num_random_blocks * config.block_size "
                f"= {max_tokens_to_attend} with config.block_size "
                f"= {self.config.block_size}, config.num_random_blocks "
                f"= {self.config.num_random_blocks}. "
                "Changing attention type to 'original_full'..."
            )
            self.set_attention_type("original_full")

        if self.attention_type == "block_sparse":
            (
                padding_len,
                input_ids,
                attention_mask,
                token_type_ids,
                position_ids,
                inputs_embeds,
            ) = self._pad_to_block_size(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                pad_token_id=self.config.pad_token_id,
            )
        else:
            padding_len = 0

        if self.attention_type == "block_sparse":
            blocked_encoder_mask, band_mask, from_mask, to_mask = self.create_masks_for_block_sparse_attn(
                attention_mask, self.block_size
            )
            extended_attention_mask = None

        elif self.attention_type == "original_full":
            blocked_encoder_mask = None
            band_mask = None
            from_mask = None
            to_mask = None
            # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
            # ourselves in which case we just need to make it broadcastable to all heads.
            extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
                attention_mask, input_shape, device
            )
        else:
            raise ValueError(
                f"attention_type can either be original_full or block_sparse, but is {self.attention_type}"
            )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # =================
        #embedding_output = self.embeddings(
        #    input_ids=input_ids,
        #    position_ids=position_ids,
        #    token_type_ids=token_type_ids,
        #    inputs_embeds=inputs_embeds,
        #    past_key_values_length=past_key_values_length,
        #)
        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        if hasattr(self, "embeddings_project"):
            embedding_output = self.embeddings_project(embedding_output)
        # =================

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            band_mask=band_mask,
            from_mask=from_mask,
            to_mask=to_mask,
            blocked_encoder_mask=blocked_encoder_mask,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        pooler_output = self.activation(self.pooler(sequence_output[:, 0, :])) if (self.pooler is not None) else None

        # undo padding
        if padding_len > 0:
            # unpad `sequence_output` because the calling function is expecting a length == input_ids.size(1)
            sequence_output = sequence_output[:, :-padding_len]

        if not return_dict:
            return (sequence_output, pooler_output) + encoder_outputs[1:]

        return transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooler_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class BigBroForSequenceClassification(transformers.BigBirdForSequenceClassification):
    def __init__(self, config):
        # =================
        #super().__init__(config)
        super(transformers.BigBirdForSequenceClassification, self).__init__(config)
        # =================
        self.num_labels = config.num_labels
        self.config = config
        # =================
        #self.bert = BigBirdModel(config)
        self.bert = BigBroModel(config)
        # =================
        self.classifier = transformers.models.big_bird.modeling_big_bird.BigBirdClassificationHead(config)

        self.init_weights()

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, torch.nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        # =================
        elif isinstance(module, transformers.models.roformer.modeling_roformer.RoFormerSinusoidalPositionalEmbedding):
            pass
        # =================
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class BigBroForMaskedLM(transformers.BigBirdForMaskedLM):
    def __init__(self, config):
        # =================
        #super().__init__(config)
        super(transformers.BigBirdForMaskedLM, self).__init__(config)
        # =================

        if config.is_decoder:
            logger.warning(
                "If you want to use `BigBroForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        # =================
        #self.bert = BigBirdModel(config)
        self.bert = BigBroModel(config)
        # =================
        self.cls = transformers.models.big_bird.modeling_big_bird.BigBirdOnlyMLMHead(config)

        self.init_weights()

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, torch.nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        # =================
        elif isinstance(module, transformers.models.roformer.modeling_roformer.RoFormerSinusoidalPositionalEmbedding):
            pass
        # =================
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
