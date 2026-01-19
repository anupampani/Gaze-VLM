import torch
import torch.nn.functional as FUN
from einops import rearrange
from torch import nn
from .helpers import PerceiverResampler,GazeResampler, GazeRegressor, GazeOverlayGenerator
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
        #self.gaze_regressor = GazeRegressor(dim=self.vis_dim)
        #self.gaze_overlay_generator = GazeOverlayGenerator()
        self.gazesampler = GazeResampler(dim=self.vis_dim,depth=2)
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
        gaze_x:torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        clear_conditioned_layers: bool = True,
        past_key_values=None,
        use_cache: bool = False,
    ):
        """
        Forward pass of Flamingo.

        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W) with F=1
            lang_x (torch.Tensor): Language input ids
                shape (B, T_txt)
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            labels (torch.Tensor, optional): Labels. Defaults to None.
            clear_conditioned_layers: if True, clear the conditioned layers
                once the foward pass is completed. Set this to false if the
                same set of images will be reused in another subsequent
                forward pass.
            past_key_values: pre-computed values to pass to language model.
                See past_key_values documentation in Hugging Face
                CausalLM models.
            use_cache: whether to use cached key values. See use_cache
                documentation in Hugging Face CausalLM models.
        """
        assert (
            self.lang_encoder.initialized_flamingo
        ), "Flamingo layers are not initialized. Please call `init_flamingo` first."

        assert (
            self.lang_encoder._use_cached_vision_x or vision_x is not None
        ), "Must provide either vision_x or have precached media using cache_media()."


        #print("Shape of vision_x before encoding",vision_x.shape)

        attn_weights =[]
        cosine_sim = 0
        if self.lang_encoder._use_cached_vision_x:
            # Case: use cached; vision_x should be cached and other
            # vision-related inputs should not be provided.
            assert (
                vision_x is None
            ), "Expect vision_x to be None when media has been cached using cache_media(). Try uncache_media() first."
            assert self.lang_encoder.is_conditioned()

        else:
            # Case: do not use caching (i.e. this is a standard forward pass);
            # changed self._encode_vision_x(vision_x=vision_x)
            #attn_weights ,cosine_sim= self._encode_vision_gaze3(vision_x=vision_x, gaze_x=gaze_x)
            attn_weights = self._encode_vision_gaze(vision_x=vision_x,gaze_x=gaze_x)
            #attn_weights = self._encode_vision_gaze(vision_x=vision_x)
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

        return output, attn_weights,cosine_sim
        #return output, attn_weights

    def generate(
        self,
        vision_x: torch.Tensor,
        gaze_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        **kwargs,
    ):
        """
        Generate text conditioned on vision and language inputs.

        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                images in the same chunk are collated along T_img, and frames are collated along F
                currently only F=1 is supported (single-frame videos)
            lang_x (torch.Tensor): Language input
                shape (B, T_txt)
            **kwargs: see generate documentation in Hugging Face CausalLM models. Some notable kwargs:
                max_length (int, optional): Maximum length of the output. Defaults to None.
                attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
                num_beams (int, optional): Number of beams. Defaults to 1.
                max_new_tokens (int, optional): Maximum new tokens. Defaults to None.
                temperature (float, optional): Temperature. Defaults to 1.0.
                top_k (int, optional): Top k. Defaults to 50.
                top_p (float, optional): Top p. Defaults to 1.0.
                no_repeat_ngram_size (int, optional): No repeat ngram size. Defaults to 0.
                length_penalty (float, optional): Length penalty. Defaults to 1.0.
                num_return_sequences (int, optional): Number of return sequences. Defaults to 1.
                do_sample (bool, optional): Do sample. Defaults to False.
                early_stopping (bool, optional): Early stopping. Defaults to False.
        Returns:
            torch.Tensor: lang_x with generated tokens appended to it
        """
        
        num_beams = kwargs.pop("num_beams", 1)
        if num_beams > 1:
            vision_x = vision_x.repeat_interleave(num_beams, dim=0)

        self.lang_encoder._use_cached_vision_x = True
        # changed self._encode_vision_x(vision_x=vision_x)
        #added 
        #attn_weights,cosine_sim = self._encode_vision_gaze2(vision_x=vision_x, gaze_x=gaze_x)
        attn_weights = self._encode_vision_gaze(vision_x=vision_x,gaze_x=gaze_x)
        #attn_weights = self._encode_vision_gaze(vision_x=vision_x)
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
        #return output,attn_weights,cosine_sim
        return output, attn_weights




    def _encode_vision_gaze(self, vision_x: torch.Tensor,gaze_x:torch.Tensor):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """

        
        assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
        b, T, F = vision_x.shape[:3]
        assert F == 1, "Only single frame supported"

        vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")  
        

        assert gaze_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
        b, T, F = gaze_x.shape[:3]
        assert F == 1, "Only single frame supported"

        gaze_x = rearrange(gaze_x, "b T F c h w -> (b T F) c h w")


        with torch.no_grad():
            vision_x = self.vision_encoder(vision_x)[1]
            gaze_x = self.vision_encoder(gaze_x)[1]
        # print("Shape of vision x before rearrange ",vision_x.shape)
        vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)
        gaze_x = rearrange(gaze_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)

        #this is where we can add the attention layer 
        attn_weights,gaze_enhanced= self.gazesampler(vision_x,gaze_x)
        combined_x = self.perceiver(gaze_enhanced)

        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(combined_x)

        return attn_weights


    # def _encode_vision_gaze(self, vision_x: torch.Tensor):
    #     assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
    #     b, T, F = vision_x.shape[:3]
    #     assert F == 1, "Only single frame supported"

    #     vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")

    #     with torch.no_grad():
    #         vision_x = self.vision_encoder(vision_x)[1]  # (b*T, v, d)

    #     # Regress gaze from ViT embeddings using transformer decoder
    #     vision_x2= vision_x
    #     vision_x2= rearrange(vision_x2,"(b T) v d -> b T v d", b=b, T=T)
    #     gaze_x = self.gaze_regressor(vision_x2)  # (b, T, 1, d)
    #     vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)



    #     # Use the regressed gaze for resampling
    #     attn_weights, gaze_enhanced = self.gazesampler(vision_x, gaze_x)
    #     combined_x = self.perceiver(gaze_enhanced)

    #     for layer in self.lang_encoder._get_decoder_layers():
    #         layer.condition_vis_x(combined_x)

    #     return attn_weights


    # def _encode_vision_gaze2(self, vision_x: torch.Tensor, gaze_x: torch.Tensor):
    #     assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
    #     b, T, F = vision_x.shape[:3]
    #     assert F == 1, "Only single frame supported"

    #     # Rearrange vision input
    #     vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
    #     gaze_x = rearrange(gaze_x, "b T F c h w -> (b T F) c h w")

    #     #print("shape of vision_x is ",vision_x.shape)

    #     # Generate gaze overlay
    #     generated_gaze_x = self.gaze_overlay_generator(vision_x)  # Output shape: (B, 1, H, W)
        
    #     # Expand generated_gaze_x channels to match vision_x
    #     #generated_gaze_x = generated_gaze_x.expand(-1, vision_x.shape[1], -1, -1)
       

    #     generated_gaze_x = generated_gaze_x.repeat(1, vision_x.shape[1], 1, 1)  # Match channel dimension
    #     #print("shape of generated is ",generated_gaze_x.shape)


    #     # Combine vision_x and generated_gaze_x
    #     #combined_gaze_x = torch.cat([vision_x, generated_gaze_x], dim=1)  # Concatenate along channels
    
    #     #print("shape of combined is ",generated_gaze_x.shape)
    #     combined_gaze_x = vision_x + generated_gaze_x
    #     combined_gaze_x = (combined_gaze_x - combined_gaze_x.min()) / (combined_gaze_x.max() - combined_gaze_x.min())
    #     # Pass combined_gaze_x through vision_encoder with no gradient computation
    #     with torch.no_grad():
    #         vision_x = self.vision_encoder(vision_x)[1]  # Extract features (b*T, v, d)
    #         combined_gaze_x = self.vision_encoder(combined_gaze_x)[1]  # Extract features (b*T, v, d)
    #         gaze_x= self.vision_encoder(gaze_x)[1]
        

    #     cosine_sim = FUN.cosine_similarity(combined_gaze_x, gaze_x, dim=-1)

    #     # Rearrange back to original shape for further processing
    #     vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)
    #     combined_gaze_x = rearrange(combined_gaze_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)

    #     # Pass through GazeResampler
    #     attn_weights, gaze_enhanced = self.gazesampler(vision_x, combined_gaze_x)
    #     combined_x = self.perceiver(gaze_enhanced)

    #     # Condition language encoder
    #     for layer in self.lang_encoder._get_decoder_layers():
    #         layer.condition_vis_x(combined_x)

    #     return attn_weights,cosine_sim.mean()



    def _encode_vision_gaze3(self, vision_x: torch.Tensor, gaze_x: torch.Tensor):
        assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
        b, T, F = vision_x.shape[:3]
        assert F == 1, "Only single frame supported"

        # Assuming vision_x and gaze_x are both in shape [b, T, F, c, h, w]
        # Flatten temporal and feature dimensions
        vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
        gaze_x = rearrange(gaze_x, "b T F c h w -> (b T F) c h w")

        # Gaze_x is a binary heatmap for each image

    # 2) Prepare saliency map (which is currently a binary heatmap)
        saliency_map = gaze_x

        # If multiple channels, collapse to a single channel
        if saliency_map.shape[1] != 1:
            # Average across channels to get a single-channel map
            saliency_map = saliency_map.mean(dim=1, keepdim=True)  # [B, 1, H, W]

        # 3) Flatten (H,W) -> one dimension so we can do min/max in older PyTorch
        B, C, H, W = saliency_map.shape
        flat = saliency_map.view(B, C, -1)  # shape [B, C, H*W]

        # 3a) Compute per-image min
        saliency_min = flat.min(dim=2, keepdim=True)[0]  # [B, C, 1]
        saliency_min = saliency_min.view(B, C, 1, 1)     # reshaped for broadcasting

        # 3b) Subtract min to shift values to [0, ...]
        saliency_map = saliency_map - saliency_min

        # 3c) Compute per-image max
        flat = saliency_map.view(B, C, -1)
        saliency_max = flat.max(dim=2, keepdim=True)[0]  # [B, C, 1]
        saliency_max = saliency_max.view(B, C, 1, 1)

        # 3d) Divide by max to get [0,1] range
        saliency_map = saliency_map / (saliency_max + 1e-8)

        # 4) Repeat saliency map across 3 channels (if your model expects RGB alignment)
        saliency_map = saliency_map.repeat(1, 3, 1, 1)  # from [B, 1, H, W] -> [B, 3, H, W]

        # 5) Multiply the original image by the saliency map
        combined_gaze_x = vision_x * saliency_map  # element-wise multiplication

        # Pass combined_gaze_x through vision_encoder with no gradient computation
        with torch.no_grad():
            vision_x = self.vision_encoder(vision_x)[1]  # Extract features (b*T, v, d)
            combined_gaze_x = self.vision_encoder(combined_gaze_x)[1]  # Extract features (b*T, v, d)
            gaze_x= self.vision_encoder(gaze_x)[1]
        

        cosine_sim = FUN.cosine_similarity(combined_gaze_x, gaze_x, dim=-1)

        # Rearrange back to original shape for further processing
        vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)
        combined_gaze_x = rearrange(combined_gaze_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)

        # Pass through GazeResampler
        attn_weights, gaze_enhanced = self.gazesampler(vision_x, combined_gaze_x)
        combined_x = self.perceiver(gaze_enhanced)

        # Condition language encoder
        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(combined_x)

        return attn_weights,cosine_sim.mean()



    def _encode_vision_x(self, vision_x: torch.Tensor):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """

        assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
        b, T, F = vision_x.shape[:3]
        assert F == 1, "Only single frame supported"

        vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
        with torch.no_grad():
            vision_x = self.vision_encoder(vision_x)[1]
            #gaze_x = sefl.visino_encoder(gaze_x)[1]
        vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)

        #this is where we can add the attention layer 

        vision_x = self.perceiver(vision_x)

        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)

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

    def cache_media(self, input_ids: torch.Tensor, vision_x: torch.Tensor,gaze_x:torch.Tensor):
        """
        Pre-cache a prompt/sequence of images / text for log-likelihood evaluations.
        All subsequent calls to forward() will generate attending to the LAST
        image in vision_x.
        This is not meant to be used to cache things for generate().
        Args:
            input_ids (torch.Tensor): Language input
                shape (B, T_txt)
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)
        """
        #changed self._encode_vision_x(vision_x=vision_x)
        #added 
        attn_weights = self._encode_vision_gaze(vision_x=vision_x,gaze_x=gaze_x)
        self._condition_media_locations(input_ids=input_ids)
        self.lang_encoder._use_cached_vision_x = True

    def uncache_media(self):
        """
        Clear all conditioning.
        """
        self.lang_encoder.clear_conditioned_layers()
        self.lang_encoder._use_cached_vision_x = False
