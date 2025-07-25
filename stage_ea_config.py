from transformers.configuration_utils import PretrainedConfig
from typing import List


class StageEaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LlamaModel`]. It is used to instantiate an LLaMA
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the LLaMA-7B.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the LLaMA model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`LlamaModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        pretraining_tp (`int`, *optional*, defaults to `1`):
            Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
            document](https://huggingface.co/docs/transformers/parallelism) to understand more about it. This value is
            necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232).
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings(`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be an float greater than 1. The expected format
            is `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
            these scaling strategies behave:
            https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
            experimental feature, subject to breaking API changes in future versions.

        Example:

    ```python
    >>> from transformers import LlamaModel, LlamaConfig

    >>> # Initializing a LLaMA llama-7b style configuration
    >>> configuration = LlamaConfig()

    >>> # Initializing a model from the llama-7b style configuration
    >>> model = LlamaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "llama"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
            self,
            vocab_size=32000,
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=None,
            hidden_act="silu",
            max_position_embeddings=2560,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=True,
            pad_token_id=None,
            bos_token_id=1,
            eos_token_id=2,
            pretraining_tp=1,
            tie_word_embeddings=False,
            rope_scaling=None,
            ea_config=None,
            stage=-1,
            stage_num_hidden_layers_list=[0],
            base_model_name_or_path=None,
            has_embedding=True,
            has_draft_model=False,
            has_lm_head=True,
            **kwargs,
    ):
        max_position_embeddings = 2560
        if ea_config is not None:
            self.vocab_size = ea_config.vocab_size
            self.max_position_embeddings = ea_config.max_position_embeddings
            self.hidden_size = ea_config.hidden_size
            self.intermediate_size = ea_config.intermediate_size
            self.num_hidden_layers = ea_config.num_hidden_layers
            self.num_attention_heads = ea_config.num_attention_heads

            # for backward compatibility
            # if num_key_value_heads is None:
            #     num_key_value_heads = ea_config.num_attention_heads

            self.num_key_value_heads = ea_config.num_key_value_heads
            self.hidden_act = ea_config.hidden_act
            self.initializer_range = ea_config.initializer_range
            self.rms_norm_eps = ea_config.rms_norm_eps
            self.pretraining_tp = ea_config.pretraining_tp
            self.use_cache = ea_config.use_cache
            self.rope_scaling = ea_config.rope_scaling
            # todo: where does the rope_theta comes from?
            if hasattr(ea_config, 'rope_theta'):
                self.rope_theta = ea_config.rope_theta
            self._rope_scaling_validation()

            super().__init__(
                pad_token_id=ea_config.pad_token_id,
                bos_token_id=ea_config.bos_token_id,
                eos_token_id=ea_config.eos_token_id,
                tie_word_embeddings=ea_config.tie_word_embeddings,
                **kwargs,
            )
        else:
            self.vocab_size = vocab_size
            self.max_position_embeddings = max_position_embeddings
            self.hidden_size = hidden_size
            self.intermediate_size = intermediate_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads

            # for backward compatibility
            if num_key_value_heads is None:
                num_key_value_heads = num_attention_heads

            self.num_key_value_heads = num_key_value_heads
            self.hidden_act = hidden_act
            self.initializer_range = initializer_range
            self.rms_norm_eps = rms_norm_eps
            self.pretraining_tp = pretraining_tp
            self.use_cache = use_cache
            self.rope_scaling = rope_scaling
            self._rope_scaling_validation()

            super().__init__(
                pad_token_id=pad_token_id,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                tie_word_embeddings=tie_word_embeddings,
                **kwargs,
            )
        self.max_position_embeddings = 2560
        self.base_model_name_or_path = base_model_name_or_path
        self.has_embedding = has_embedding
        self.has_draft_model = has_draft_model
        self.has_lm_head = has_lm_head
        # assert has_embedding == has_lm_head  # since the sharing weights

        # assert isinstance(stage, int)
        # assert stage >= -1, 'Stage must be non-negative'
        self.stage = stage
        # print(type(stage_num_hidden_layers_list))

        # assert isinstance(stage_num_hidden_layers_list, list) and all(isinstance(n, int) for n in stage_num_hidden_layers_list)
        # assert sum(stage_num_hidden_layers_list) == self.num_hidden_layers

        assert stage_num_hidden_layers_list[0] == 0
        # [update]: n_split for base model
        self.n_split = sum([1 if l > 0 else 0 for l in stage_num_hidden_layers_list])
        self.total_stage = int(len(stage_num_hidden_layers_list))
        # if self.total_stage == 1:
        #     raise ValueError("total_stage cannot be 1")
        # assert self.stage < self.total_stage
        
        self.stage_num_hidden_layers_list = stage_num_hidden_layers_list
        self.num_stage_hidden_layers = self.stage_num_hidden_layers_list[self.stage]
        self.layer_range = (
            sum(stage_num_hidden_layers_list[:self.stage]),
            sum(stage_num_hidden_layers_list[:self.stage+1])
            )  # [start_layer_idx, end_layer_idx)
        
        # [update] is_draft_stage: stage 0 is draft stage
        self.is_draft_stage = self.stage == 0
        self.is_first_stage = self.stage == 1
        self.is_last_stage = self.stage == self.total_stage - 1
        self.last_rank = self.total_stage - 1 if self.stage == 0 else self.stage - 1
        self.next_rank = 0 if self.stage == self.total_stage - 1 else self.stage + 1

        # [MODIFIED] add network config
        self.master_ip = None
        self.ip = None
        self.init_method = None
        self.backend = None
        self.device = None


    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `name` and `factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s name field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be an float > 1, got {rope_scaling_factor}")
