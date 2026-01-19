import torch
from einops import rearrange
from torch import nn
from .helpers_gaze import PerceiverResampler,GazeResampler
from torch.distributed.fsdp.wrap import (
    enable_wrap,
    wrap,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)

from .utils import apply_with_stopping_condition



class Flamingo(nn.Module):
    def __init__(
        self,
        vision_encoder: nn.Module,
        lang_encoder: nn.Module,
        eoc_token_id: int,
        media_token_id: int,
        vis_dim: int,
        cross_attn_every_n_layers: int = 1,
        gradient_checkpointing: bool = False,
    ):
        """
        Args:
            vision_encoder (nn.Module): HF CLIPModel
            lang_encoder (nn.Module): HF causal language model
            eoc_token_id (int): Token id for <|endofchunk|>
            media_token_id (int): Token id for <image>
            vis_dim (int): Dimension of the visual features.
                Visual features are projected to match this shape along the last dimension.
            cross_attn_every_n_layers (int, optional): How often to apply cross attention after transformer layer. Defaults to 1.
        """
        super().__init__()
        self.eoc_token_id = eoc_token_id
        self.media_token_id = media_token_id
        self.vis_dim = vis_dim
        if hasattr(lang_encoder.config, "d_model"):
            self.lang_dim = lang_encoder.config.d_model  # mpt uses d_model
        else:
            self.lang_dim = lang_encoder.config.hidden_size
        print(f"lang dim size {self.lang_dim}")

        self.vision_encoder = vision_encoder.visual
        self.perceiver = PerceiverResampler(dim=self.vis_dim,max_num_media=5)
        self.gazesampler = GazeResampler(dim=self.vis_dim,depth=1)
        self.lang_encoder = lang_encoder
        self.lang_encoder.init_flamingo(
            media_token_id=media_token_id,
            lang_hidden_size=self.lang_dim,
            vis_hidden_size=self.vis_dim,
            cross_attn_every_n_layers=cross_attn_every_n_layers,
            gradient_checkpointing=gradient_checkpointing,
        )
        self._use_gradient_checkpointing = gradient_checkpointing
        self.perceiver._use_gradient_checkpointing = gradient_checkpointing
        self.gazesampler._use_gradient_checkpointing = gradient_checkpointing


    def forward(
        self,
        vision_x: torch.Tensor,
        gaze_x: torch.Tensor,
        gaze_proportions: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        clear_conditioned_layers: bool = True,
        past_key_values=None,
        use_cache: bool = False,
    ):
        assert self.lang_encoder.initialized_flamingo, (
            "Flamingo layers are not initialized. Please call `init_flamingo` first."
        )
        assert (
            self.lang_encoder._use_cached_vision_x or vision_x is not None
        ), "Must provide either vision_x or have precached media using cache_media()."

        if self.lang_encoder._use_cached_vision_x:
            assert (
                vision_x is None
            ), "Expect vision_x to be None when media has been cached using cache_media(). Try uncache_media() first."
            assert self.lang_encoder.is_conditioned()
        else:
            self._encode_vision_gaze(vision_x=vision_x, gaze_x=gaze_x,gaze_proportions=gaze_proportions)
            self._condition_media_locations(input_ids=lang_x)

        output = self.lang_encoder(
            input_ids=lang_x,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        if clear_conditioned_layers:
            self.lang_encoder.clear_conditioned_layers()

        return output

    def generate(
        self,
        vision_x: torch.Tensor,
        gaze_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        **kwargs,
    ):
        num_beams = kwargs.pop("num_beams", 1)
        if num_beams > 1:
            vision_x = vision_x.repeat_interleave(num_beams, dim=0)

        self.lang_encoder._use_cached_vision_x = True
        self._encode_vision_gaze(vision_x=vision_x, gaze_x=gaze_x)

        eos_token_id = kwargs.pop("eos_token_id", self.eoc_token_id)
        output = self.lang_encoder.generate(
            input_ids=lang_x,
            attention_mask=attention_mask,
            eos_token_id=eos_token_id,
            num_beams=num_beams,
            **kwargs,
        )

        self.lang_encoder.clear_conditioned_layers()
        self.lang_encoder._use_cached_vision_x = False
        return output


    def _encode_vision_gaze(self, vision_x: torch.Tensor, gaze_x: torch.Tensor, gaze_proportions: torch.Tensor):
        """
        Encodes vision and gaze inputs and applies externally computed gaze proportions.

        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
            gaze_x (torch.Tensor): Gaze input
                shape (B, T_img, F, C, H, W)
            gaze_proportions (torch.Tensor): Gaze proportions
                shape (B, T_img, F, V), where V is the number of patches
        """
        assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
        b, T, F = vision_x.shape[:3]
        assert F == 1, "Only single frame supported"

        # Flatten spatial dimensions
        vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
        gaze_x = rearrange(gaze_x, "b T F c h w -> (b T F) c h w")

        # Extract vision and gaze features using the encoder
        with torch.no_grad():
            vision_x = self.vision_encoder(vision_x)[1]  # Encoded vision features
            gaze_x = self.vision_encoder(gaze_x)[1]      # Encoded gaze features

        # Reshape back to (b, T, F, V, D)
        vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)
        gaze_x = rearrange(gaze_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)
        gaze_proportions=rearrange(gaze_proportions,"(b T) F v -> b T F v",b=b,T=T,F=F)

        # Apply gaze proportions directly to weight vision features
        gaze_weighted = vision_x * gaze_proportions.unsqueeze(-1)

        # # Apply gaze proportions directly to weight vision features
        gaze_weighted = rearrange(gaze_weighted, "b T F V D -> b T (F V) D")
        
        # Pass the weighted features through the Perceiver
        combined_x = self.perceiver(gaze_weighted)

        # Condition the language model layers with the combined visual input
        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(combined_x)


    def wrap_fsdp(self, wrapper_kwargs, device_id):
        """
        Manually wraps submodules for FSDP and move other parameters to device_id.

        Why manually wrap?
        - all parameters within the FSDP wrapper must have the same requires_grad.
            We have a mix of frozen and unfrozen parameters.
        - model.vision_encoder.visual needs to be individually wrapped or encode_vision_x errors
            See: https://github.com/pytorch/pytorch/issues/82461#issuecomment-1269136344

        The rough wrapping structure is:
        - FlamingoModel
            - FSDP(FSDP(vision_encoder))
            - FSDP(FSDP(perceiver))
            - lang_encoder
                - FSDP(FSDP(input_embeddings))
                - FlamingoLayers
                    - FSDP(FSDP(gated_cross_attn_layer))
                    - FSDP(FSDP(decoder_layer))
                - FSDP(FSDP(output_embeddings))
                - other parameters

        Known issues:
        - Our FSDP strategy is not compatible with tied embeddings. If the LM embeddings are tied,
            train with DDP or set the --freeze_lm_embeddings flag to true.
        - With FSDP + gradient ckpting, one can increase the batch size with seemingly no upper bound.
            Although the training curves look okay, we found that downstream performance dramatically
            degrades if the batch size is unreasonably large (e.g., 100 MMC4 batch size for OPT-125M).

        FAQs about our FSDP wrapping strategy:
        Why double wrap?
        As of torch==2.0.1, FSDP's _post_forward_hook and _post_backward_hook
        only free gathered parameters if the module is NOT FSDP root.

        Why unfreeze the decoder_layers?
        See https://github.com/pytorch/pytorch/issues/95805
        As of torch==2.0.1, FSDP's _post_backward_hook is only registed if the flat param
        requires_grad=True. We need the postback to fire to avoid OOM.
        To effectively freeze the decoder layers, we exclude them from the optimizer.

        What is assumed to be frozen v. unfrozen?
        We assume that the model is being trained under normal Flamingo settings
        with these lines being called in factory.py:
            ```
            # Freeze all parameters
            model.requires_grad_(False)
            assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0

            # Unfreeze perceiver, gated_cross_attn_layers, and LM input embeddings
            model.perceiver.requires_grad_(True)
            model.lang_encoder.gated_cross_attn_layers.requires_grad_(True)
            [optional] model.lang_encoder.get_input_embeddings().requires_grad_(True)
            ```
        """
        # unfreeze the decoder layers
        for block in self.lang_encoder.old_decoder_blocks:
            block.requires_grad_(True)

        # wrap in FSDP
        with enable_wrap(wrapper_cls=FSDP, **wrapper_kwargs):
            self.perceiver = wrap(wrap(self.perceiver))
            self.gazesampler = wrap(wrap(self.gazesampler))

            self.lang_encoder.old_decoder_blocks = nn.ModuleList(
                wrap(wrap(block)) for block in self.lang_encoder.old_decoder_blocks
            )
            self.lang_encoder.gated_cross_attn_layers = nn.ModuleList(
                wrap(wrap(layer)) if layer is not None else None
                for layer in self.lang_encoder.gated_cross_attn_layers
            )
            self.lang_encoder.init_flamingo_layers(self._use_gradient_checkpointing)
            # self.lang_encoder.set_input_embeddings(
            #     wrap(wrap(self.lang_encoder.get_input_embeddings()))
            # )
            # self.lang_encoder.set_output_embeddings(
            #     wrap(wrap(self.lang_encoder.get_output_embeddings()))
            # )
            self.vision_encoder = wrap(wrap(self.vision_encoder))  # frozen

        # manually move non-FSDP managed parameters to device_id
        # these are all in lang_encoder
        apply_with_stopping_condition(
            module=self.lang_encoder,
            apply_fn=lambda m: m.to(device_id),
            apply_condition=lambda m: len(list(m.children())) == 0,
            stopping_condition=lambda m: isinstance(m, FSDP),
        )

        # exclude the original decoder layers from the optimizer
        for block in self.lang_encoder.old_decoder_blocks:
            for p in block.parameters():
                p.exclude_from_optimizer = True

        # set up clip_grad_norm_ function
        def clip_grad_norm_(max_norm):
            self.perceiver.clip_grad_norm_(max_norm)
            for layer in self.lang_encoder.gated_cross_attn_layers:
                if layer is not None:
                    layer.clip_grad_norm_(max_norm)
            # self.lang_encoder.get_input_embeddings().clip_grad_norm_(max_norm)

        self.clip_grad_norm_ = clip_grad_norm_

    def _condition_media_locations(self, input_ids: torch.Tensor):
        """
        Compute the media token locations from lang_x and condition the language model on these.
        Args:
            input_ids (torch.Tensor): Language input
                shape (B, T_txt)
        """
        media_locations = input_ids == self.media_token_id

        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_media_locations(media_locations)    


    def cache_media(self, input_ids: torch.Tensor, vision_x: torch.Tensor, gaze_x: torch.Tensor):
        self._encode_vision_gaze(vision_x=vision_x, gaze_x=gaze_x)
        self._condition_media_locations(input_ids=input_ids)
        self.lang_encoder._use_cached_vision_x = True

    def uncache_media(self):
        """
        Clear all conditioning.
        """
        self.lang_encoder.clear_conditioned_layers()
        self.lang_encoder._use_cached_vision_x = False
