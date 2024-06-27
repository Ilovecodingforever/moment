import logging
import warnings
from argparse import Namespace
from copy import deepcopy
from math import ceil

import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn
from transformers import T5Config, T5EncoderModel, T5Model
from transformers.models.t5.modeling_t5 import T5Stack

from momentfm.common import TASKS
from momentfm.data.base import TimeseriesOutputs
from momentfm.models.layers.embed import PatchEmbedding, Patching
from momentfm.models.layers.revin import RevIN
from momentfm.utils.masking import Masking
from momentfm.utils.utils import (
    NamespaceWithDefaults,
    get_anomaly_criterion,
    get_huggingface_model_dimensions,
)

SUPPORTED_HUGGINGFACE_MODELS = [
    "google/flan-t5-small",
    "google/flan-t5-base",
    "google/flan-t5-large",
    "google/flan-t5-xl",
    "google/flan-t5-xxl",
]



##############################

from momentfm.models.t5_with_prefix import T5ForConditionalGenerationWithPrefix, T5WithPrefixConfig, T5StackWithPrefix
from momentfm.models.t5_multivariate_prefix import T5StackWithPrefixMulti, T5ForConditionalGenerationWithPrefixMulti



class MPT(nn.Module):
    """
    multitask prompt tuning
    """
    def __init__(self,
                wte: nn.Module,
                tasks: list,
                n_tokens: int = 10,
                random_range: float = 0.5,
                hidden_size: int = 64):
        """appends learned embedding to

        Args:
            wte (nn.Embedding): original transformer word embedding
            n_tokens (int, optional): number of tokens for task. Defaults to 10.
            random_range (float, optional): range to init embedding (if not initialize from vocab).
                            Defaults to 0.5.
            initialize_from_vocab (bool, optional): initalizes from default vocab. Defaults to True.
        """
        super().__init__()
        self.wte = wte
        self.n_tokens = n_tokens

        self.hidden_size = hidden_size

        self.mlp = nn.Linear(self.hidden_size, wte.weight.size(0))

        self.shared_prompt = nn.parameter.Parameter(self.initialize_embedding(n_tokens,
                                                                               self.hidden_size,
                                                                               random_range))

        self.tasks = tasks
        self.task_prompt = nn.ParameterDict({task: nn.ParameterDict({
            'u': nn.parameter.Parameter(self.initialize_embedding(n_tokens,
                                                                    1,
                                                                    random_range)),
            'v': nn.parameter.Parameter(self.initialize_embedding(1,
                                                                    self.hidden_size,
                                                                    random_range)),
            }) for task in tasks})

        self.task_name = None


    def initialize_embedding(self,
                             i: int,
                             j: int,
                             random_range: float = 0.5, ):
        """initializes learned embedding
        Args:
            same as __init__
        Returns:
            torch.float: initialized using original schemes
        """
        return torch.FloatTensor(i, j).uniform_(-random_range, random_range)


    def forward(self, tokens):
        """run forward pass
        Args:
            tokens (torch.long): input tokens before encoding
        Returns:
            torch.float: encoding of text concatenated with learned task specifc embedding
        """

        task = self.task_name

        input_embedding = self.wte(tokens)

        u = self.task_prompt[task]['u']
        v = self.task_prompt[task]['v']

        learned_embedding = self.mlp(torch.matmul(u, v) * self.shared_prompt)

        # n_batches x features x n_patches x embedding_size
        learned_embedding = learned_embedding.repeat(input_embedding.size(0),
                                                     input_embedding.size(1), 1, 1)

        self.task_name = None

        return torch.cat([learned_embedding, input_embedding], 2)

##############################



class PretrainHead(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        patch_len: int = 8,
        head_dropout: float = 0.1,
        orth_gain: float = 1.41,
    ):
        super().__init__()
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(d_model, patch_len)

        if orth_gain is not None:
            torch.nn.init.orthogonal_(self.linear.weight, gain=orth_gain)
            self.linear.bias.data.zero_()

    def forward(self, x):
        x = self.linear(self.dropout(x))
        x = x.flatten(start_dim=2, end_dim=3)
        return x


class ForecastingHead(nn.Module):
    def __init__(
        self, head_nf: int = 768 * 64, forecast_horizon: int = 96, head_dropout: int = 0
    ):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(head_nf, forecast_horizon)

    def forward(self, x, input_mask: torch.Tensor = None):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class MOMENT(nn.Module):
    def __init__(self, config: Namespace | dict, **kwargs: dict):
        super().__init__()
        config = self._update_inputs(config, **kwargs)
        config = self._validate_inputs(config)
        self.config = config
        self.task_name = config.task_name
        self.seq_len = config.seq_len
        self.patch_len = config.patch_len

        self.normalizer = RevIN(
            num_features=1, affine=config.getattr("revin_affine", False)
        )
        self.tokenizer = Patching(
            patch_len=config.patch_len, stride=config.patch_stride_len
        )
        self.patch_embedding = PatchEmbedding(
            d_model=config.d_model,
            seq_len=config.seq_len,
            patch_len=config.patch_len,
            stride=config.patch_stride_len,
            dropout=config.getattr("dropout", 0.1),
            add_positional_embedding=config.getattr("add_positional_embedding", True),
            value_embedding_bias=config.getattr("value_embedding_bias", False),
            orth_gain=config.getattr("orth_gain", 1.41),
        )
        self.mask_generator = Masking(mask_ratio=config.getattr("mask_ratio", 0.0))
        self.encoder = self._get_transformer_backbone(config)
        ###################################
        # multi-task prompt tuning
        if config.getattr("MPT", False):
            self.patch_embedding.value_embedding = MPT(self.patch_embedding.value_embedding,
                                                        config.task_names,
                                                        n_tokens=config.getattr("num_prefix", 2),)

        self.recon_head = self._get_head(TASKS.RECONSTRUCTION)
        self.fore_head = self._get_head(TASKS.FORECASTING)
        self.emb_head = self._get_head(TASKS.EMBED)
        ###################################

        # Frozen parameters
        self.freeze_embedder = config.getattr("freeze_embedder", True)
        self.freeze_encoder = config.getattr("freeze_encoder", True)
        self.freeze_head = config.getattr("freeze_head", False)

        if self.freeze_embedder:
            self.patch_embedding = freeze_parameters(self.patch_embedding)
        if self.freeze_encoder:
            self.encoder = freeze_parameters(self.encoder)
        if self.freeze_head:
            self.head = freeze_parameters(self.head)

    def _update_inputs(
        self, config: Namespace | dict, **kwargs: dict
    ) -> NamespaceWithDefaults:
        if isinstance(config, dict) and "model_kwargs" in kwargs:
            return NamespaceWithDefaults(**{**config, **kwargs["model_kwargs"]})
        else:
            return NamespaceWithDefaults.from_namespace(config)

    def _validate_inputs(self, config: NamespaceWithDefaults) -> NamespaceWithDefaults:
        if (
            config.d_model is None
            and config.transformer_backbone in SUPPORTED_HUGGINGFACE_MODELS
        ):
            config.d_model = get_huggingface_model_dimensions(
                config.transformer_backbone
            )
            logging.info(f"Setting d_model to {config.d_model}")
        elif config.d_model is None:
            raise ValueError(
                "d_model must be specified if transformer backbone "
                "unless transformer backbone is a Huggingface model."
            )

        if config.transformer_type not in [
            "encoder_only",
            "decoder_only",
            "encoder_decoder",
        ]:
            raise ValueError(
                "transformer_type must be one of "
                "['encoder_only', 'decoder_only', 'encoder_decoder']"
            )

        if config.patch_stride_len != config.patch_len:
            warnings.warn("Patch stride length is not equal to patch length.")
        return config

    def _get_head(self, task_name: str) -> nn.Module:
        if task_name == TASKS.RECONSTRUCTION:
            return PretrainHead(
                self.config.d_model,
                self.config.patch_len,
                self.config.getattr("dropout", 0.1),
                self.config.getattr("orth_gain", 1.41),
            )
        elif task_name == TASKS.FORECASTING:
            num_patches = (
                max(self.config.seq_len, self.config.patch_len) - self.config.patch_len
            ) // self.config.patch_stride_len + 1
            self.head_nf = self.config.d_model * (num_patches + (self.config.num_prefix if self.config.getattr("MPT", False) else 0))
            return ForecastingHead(
                self.head_nf,
                self.config.forecast_horizon,
                self.config.getattr("head_dropout", 0.1),
            )
        elif task_name == TASKS.EMBED:
            return nn.Identity()
        else:
            raise NotImplementedError(f"Task {task_name} not implemented.")

    def _get_transformer_backbone(self, config) -> nn.Module:
        if config.getattr("randomly_initialize_backbone", False):
            model_config = T5Config.from_pretrained(config.transformer_backbone)
            transformer_backbone = T5Model(model_config)
            logging.info(
                f"Initializing randomly initialized transformer from {config.transformer_backbone}."
            )
        else:
            transformer_backbone = T5EncoderModel.from_pretrained(
                config.transformer_backbone
            )
            logging.info(
                f"Initializing pre-trained transformer from {config.transformer_backbone}."
            )

        transformer_backbone = transformer_backbone.get_encoder()

        model_config = transformer_backbone.config

        # if config.getattr("enable_gradient_checkpointing", True):
        #     transformer_backbone.gradient_checkpointing_enable()
        #     logging.info("Enabling gradient checkpointing.")

        #######################################################################
        # prefix tuning
        if config.getattr("prefix_tuning", False) or config.getattr("prefix_tuning_multi", False):
            logging.info("Using prefix tuning.")

            model_config = T5Config.from_pretrained(config.transformer_backbone)
            setattr(model_config, 'num_prefix', self.config.getattr("num_prefix", 2))
            setattr(model_config, 'reparam', True)
            setattr(model_config, 'reparam_dim', 32)
            setattr(model_config, 'no_decoder_self_attn', False) # TODO
            setattr(model_config, 'MPT', self.config.getattr("MPT", False))
            setattr(model_config, 'seq_len', self.config.seq_len)

            num_patches = (max(self.config.seq_len, self.config.patch_len) - self.config.patch_len
                        ) // self.config.patch_stride_len + 1
            setattr(model_config, 'num_patches', num_patches)


            if config.getattr("prefix_tuning_multi", False):
                setattr(model_config, 'prefix_tuning', config.getattr("prefix_tuning", False))
                transformer_backbone = T5ForConditionalGenerationWithPrefixMulti(model_config)

            elif config.getattr("prefix_tuning", False):
                transformer_backbone = T5ForConditionalGenerationWithPrefix(model_config)
            transformer_backbone = transformer_backbone.from_pretrained(config.transformer_backbone, config=model_config)
            
            transformer_backbone.enable_input_require_grads()
            transformer_backbone = transformer_backbone.encoder

            # check whether the weights is loaded correctly
            # t5 = T5EncoderModel.from_pretrained(config.transformer_backbone).get_encoder()
            # for name, param in transformer_backbone.named_parameters():
            #     if 'prefix' not in name and 'prompt' not in name:
            #         assert(torch.equal(param, t5.state_dict()[name]))

            # gradient checkpointing for prefix tuning doesn't work:
            # past_key_value is always None with gradient checkpointing (from T5Stack source code)
        #######################################################################

        return transformer_backbone

    def __call__(self, *args, **kwargs) -> TimeseriesOutputs:
        return self.forward(*args, **kwargs)

    def embed(
        self,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        reduction: str = "mean",
        **kwargs,
    ) -> TimeseriesOutputs:
        batch_size, n_channels, seq_len = x_enc.shape

        if input_mask is None:
            input_mask = torch.ones((batch_size, seq_len)).to(x_enc.device)

        x_enc = self.normalizer(x=x_enc, mask=input_mask, mode="norm")
        x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)

        input_mask_patch_view = Masking.convert_seq_to_patch_view(
            input_mask, self.patch_len
        )

        x_enc = self.tokenizer(x=x_enc)

        enc_in = self.patch_embedding(x_enc, mask=input_mask)

        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape(
            (batch_size * n_channels, n_patches, self.config.d_model)
        )

        patch_view_mask = Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
        attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0)
        outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
        enc_out = outputs.last_hidden_state

        enc_out = enc_out.reshape((-1, n_channels, n_patches, self.config.d_model))
        # [batch_size x n_channels x n_patches x d_model]

        if reduction == "mean":
            enc_out = enc_out.mean(dim=1, keepdim=False)  # Mean across channels
            # [batch_size x n_patches x d_model]
            input_mask_patch_view = input_mask_patch_view.unsqueeze(-1).repeat(
                1, 1, self.config.d_model
            )
            enc_out = (input_mask_patch_view * enc_out).sum(
                dim=1
            ) / input_mask_patch_view.sum(dim=1)
        else:
            raise NotImplementedError(f"Reduction method {reduction} not implemented.")

        return TimeseriesOutputs(
            embeddings=enc_out, input_mask=input_mask, metadata=reduction
        )

    def reconstruction(
        self,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        mask: torch.Tensor = None,
        task_name: str = None,
        **kwargs,
    ) -> TimeseriesOutputs:
        batch_size, n_channels, _ = x_enc.shape

        if mask is None:
            mask = self.mask_generator.generate_mask(x=x_enc, input_mask=input_mask)
            mask = mask.to(x_enc.device)  # mask: [batch_size x seq_len]

        x_enc = self.normalizer(x=x_enc, mask=mask * input_mask, mode="norm")
        # Prevent too short time-series from causing NaNs
        x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)

        x_enc = self.tokenizer(x=x_enc)

        #######################################################################
        if isinstance(self.patch_embedding.value_embedding, MPT):

            # using multitask prompt tuning
            # use 0 or 1? Used 0 in multivariate prefix tuning
            # should use 1. https://discuss.huggingface.co/t/clarification-on-the-attention-mask/1538
            # In multivariate prefix tuning, 1 and 0 are already inverted
            mask = torch.concatenate([torch.ones(mask.size(0),
                                                self.patch_len*self.patch_embedding.value_embedding.n_tokens).to(x_enc.device),
                                        mask], dim=1)
            input_mask = torch.concatenate([torch.ones(input_mask.size(0),
                                                        self.patch_len*self.patch_embedding.value_embedding.n_tokens).to(x_enc.device),
                                            input_mask], dim=1)

            self.patch_embedding.value_embedding.task_name = task_name

        # #######################################################################

        enc_in = self.patch_embedding(x_enc, mask=mask)

        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape(
            (batch_size * n_channels, n_patches, self.config.d_model)
        )

        patch_view_mask = Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
        attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0)


        if self.config.transformer_type == "encoder_decoder":
            # outputs = self.encoder(
            #     inputs_embeds=enc_in,
            #     decoder_inputs_embeds=enc_in,
            #     attention_mask=attention_mask,
            # )
            raise NotImplementedError("Encoder-decoder not implemented for prefix T5.")
        else:
            outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask,)


        enc_out = outputs.last_hidden_state

        enc_out = enc_out.reshape((-1, n_channels, n_patches, self.config.d_model))

        #######################################################################
        dec_out = self.recon_head(enc_out)  # [batch_size x n_channels x seq_len]
        #######################################################################

        dec_out = self.normalizer(x=dec_out, mode="denorm")

        if self.config.getattr("debug", False):
            illegal_output = self._check_model_weights_for_illegal_values()
        else:
            illegal_output = None

        #######################################################################
        # if using prompt tuning
        if not isinstance(self.patch_embedding.value_embedding, nn.Linear):
            dec_out = dec_out[:, :, self.patch_len*self.patch_embedding.value_embedding.n_tokens:]

        # get embedding
        input_mask_patch_view = Masking.convert_seq_to_patch_view(
            input_mask, self.patch_len
        )
        enc_out = enc_out.mean(dim=1, keepdim=False)  # Mean across channels
        # [batch_size x n_patches x d_model]
        input_mask_patch_view = input_mask_patch_view.unsqueeze(-1).repeat(
            1, 1, self.config.d_model
        )
        enc_out = (input_mask_patch_view * enc_out).sum(
            dim=1
        ) / input_mask_patch_view.sum(dim=1)
        #######################################################################


        return TimeseriesOutputs(
            input_mask=input_mask,
            reconstruction=dec_out,
            pretrain_mask=mask,
            illegal_output=illegal_output,
        #######################################################################
            embeddings=enc_out
        #######################################################################
        )

    def forecast(
        self, x_enc: torch.Tensor, input_mask: torch.Tensor = None,
        task_name: str = None,
        **kwargs
    ) -> TimeseriesOutputs:
        batch_size, n_channels, seq_len = x_enc.shape

        x_enc = self.normalizer(x=x_enc, mask=input_mask, mode="norm")
        x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)

        x_enc = self.tokenizer(x=x_enc)

        #######################################################################
        if isinstance(self.patch_embedding.value_embedding, MPT):
            # using prompt tuning
            input_mask = torch.concatenate([torch.ones(input_mask.size(0),
                                                       self.patch_len*self.patch_embedding.value_embedding.n_tokens).to(x_enc.device),
                                            input_mask], dim=1)

            self.patch_embedding.value_embedding.task_name = task_name

        #######################################################################

        enc_in = self.patch_embedding(x_enc, mask=torch.ones_like(input_mask))

        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape(
            (batch_size * n_channels, n_patches, self.config.d_model)
        )

        patch_view_mask = Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
        attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0)
        outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
        enc_out = outputs.last_hidden_state
        enc_out = enc_out.reshape((-1, n_channels, n_patches, self.config.d_model))
        # [batch_size x n_channels x n_patches x d_model]

        #######################################################################
        dec_out = self.fore_head(enc_out)  # [batch_size x n_channels x forecast_horizon]
        #######################################################################
        dec_out = self.normalizer(x=dec_out, mode="denorm")

        return TimeseriesOutputs(input_mask=input_mask, forecast=dec_out)


    def forward(
        self,
        x_enc: torch.Tensor,
        mask: torch.Tensor = None,
        input_mask: torch.Tensor = None,
        task_name: str = None,
        **kwargs,
    ) -> TimeseriesOutputs:
        if input_mask is None:
            input_mask = torch.ones_like(x_enc[:, 0, :])

        if self.task_name == TASKS.RECONSTRUCTION:
            return self.reconstruction(
                x_enc=x_enc, mask=mask, input_mask=input_mask,
                task_name=task_name,
                **kwargs
            )
        elif self.task_name == TASKS.EMBED:
            return self.embed(x_enc=x_enc, input_mask=input_mask, **kwargs)
        elif self.task_name == TASKS.FORECASTING:
            return self.forecast(x_enc=x_enc, input_mask=input_mask,
                                 task_name=task_name,
                                 **kwargs)
        else:
            raise NotImplementedError(f"Task {self.task_name} not implemented.")


class MOMENTPipeline(MOMENT, PyTorchModelHubMixin):
    def __init__(self, config: Namespace | dict, **kwargs: dict):
        self._validate_model_kwargs(**kwargs)
        self.new_task_name = kwargs.get("model_kwargs", {}).pop(
            "task_name", TASKS.RECONSTRUCTION
        )
        super().__init__(config, **kwargs)

    def _validate_model_kwargs(self, **kwargs: dict) -> None:
        kwargs = deepcopy(kwargs)
        kwargs.setdefault("model_kwargs", {"task_name": TASKS.RECONSTRUCTION})
        kwargs["model_kwargs"].setdefault("task_name", TASKS.RECONSTRUCTION)
        config = Namespace(**kwargs["model_kwargs"])

        if config.task_name == TASKS.FORECASTING:
            if not hasattr(config, "forecast_horizon"):
                raise ValueError(
                    "forecast_horizon must be specified for long-horizon forecasting."
                )

    def init(self) -> None:
        if self.new_task_name != TASKS.RECONSTRUCTION:
            self.task_name = self.new_task_name
            self.head = self._get_head(self.new_task_name)

def freeze_parameters(model):
    """
    Freeze parameters of the model
    """
    # Freeze the parameters
    for name, param in model.named_parameters():
        param.requires_grad = False

    return model
