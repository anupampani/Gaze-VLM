"""
Based on: https://github.com/lucidrains/flamingo-pytorch
"""

import torch
from einops import rearrange, repeat
from einops_exts import rearrange_many
from torch import einsum, nn
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def exists(val):
    return val is not None


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


class CustomQueryTransform(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CustomQueryTransform, self).__init__()
        self.linear_current = nn.Linear(input_dim, output_dim, bias=False)
        self.linear_cumulative = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, queries):
        b, T, _, d = queries.shape
        transformed_queries = []
        cumulative_transformed_query = torch.zeros(b, d).to(queries.device)
        for t in range(T):
            if t == 0:
                transformed_query = self.linear_current(queries[:, t, 0, :])
            else:
                cumulative_transformed_query += transformed_queries[-1]
                transformed_query = self.linear_current(queries[:, t, 0, :]) + self.linear_cumulative(cumulative_transformed_query)
            transformed_queries.append(transformed_query)
        return torch.stack(transformed_queries, dim=1).unsqueeze(2) 



class CustomQueryTransformLSTM(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CustomQueryTransformLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, output_dim, batch_first=True)

    def forward(self, queries):
        b, T, _, d = queries.shape
        queries = queries.squeeze(2)  # Remove the extra dimension to match LSTM input shape (b, T, d)
        lstm_out, _ = self.lstm(queries)
        return lstm_out.unsqueeze(2)  # Add back the dimension

class GazeResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=2,
        dim_head=64,
        heads=8,
        num_latents=1,
        ff_mult=4,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        #GazeAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )
        #self.query_transform= CustomQueryTransformLSTM(dim,dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, gaze_proportions):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, F, v, D)
            gaze_proportions (torch.Tensor): gaze proportions
                shape (b, T, F, v)
        Returns:
            shape (b, T, n, D) where n is self.num_latents
        """
        b, T, F, v = x.shape[:4]

        # Flatten frame and spatial dimensions
        x = rearrange(x, "b T F v d -> b T (F v) d")
        gaze_proportions = rearrange(gaze_proportions, "b T F v -> b T (F v)")

        # Normalize gaze proportions across spatial dimensions
        gaze_proportions = F.normalize(gaze_proportions, p=1, dim=-1)

        # Apply gaze proportions as weights to vision features
        gaze_weighted_x = x * gaze_proportions.unsqueeze(-1)  # Broadcast proportions over feature dimension

        for ff in self.layers:
            gaze_weighted_x = ff(gaze_weighted_x) + gaze_weighted_x

        return self.norm(gaze_weighted_x)


    # def forward(self, x,gaze_proportion):
    #     """
    #     Args:
    #         x (torch.Tensor): image features
    #             shape (b, T, F, v, D)
    #     Returns:
    #         shape (b, T, n, D) where n is self.num_latents
    #     """
    #     b, T, F, v = x.shape[:4]

    #     x = rearrange(
    #         x, "b T F v d -> b T (F v) d"
    #     )  # flatten the frame and spatial dimensions


    #     gaze_x = rearrange(
    #         gaze_x, "b T F v d -> b T (F v) d"
    #     )  # flatten the frame and spatial dimensions


    #     gaze_x= gaze_x.mean(dim=2,keepdim=True)
    #    # gaze_x = self.query_transform(gaze_x)
    #     for attn, ff in self.layers:
    #         attn_weights,attn_score= attn(x,gaze_x)

    #         gaze_x = attn_score + gaze_x
    #         gaze_x = ff(gaze_x) + gaze_x
    #     return attn_weights,self.norm(gaze_x)


class GazeAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8,patches=256,n_queries=1):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads
        

        self.norm_media = nn.LayerNorm(dim)
        self.norm_gaze = nn.LayerNorm(dim)
        

        self.q_init= nn.Linear(patches,n_queries,bias=False)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)  # Gaze features as queries
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)  # Vision features as keys and values
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        #self.to_attention = nn.Linear(2048,256,bias=False)

    def forward(self, vision_x, gaze_x):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, n1, D)

        """

        vision_x = self.norm_media(vision_x)
        q = self.norm_gaze(gaze_x)
        h = self.heads


        q = self.to_q(gaze_x)
        # q= q.permute(0,1,3,2)
        # q= self.q_init(q)
        # q= q.permute(0,1,3,2)

        kv_input = vision_x
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        #getting rid of multi-head attention- not required and makes things complex 
        # q, k, v = rearrange_many((q, k, v), "b t n (h d) -> b h t n d", h=h)
        q = q * self.scale
        # print("shape of query is ",q.shape)
        # print("shape of key and value is ",k.shape)

        # attention
        sim = einsum("... i d, ... j d  -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        # print("shape of attention weights is ",attn.shape)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        # out = rearrange(out, "b h t n d -> b t n (h d)", h=h)
        return attn,self.to_out(out)




    def forward_multi_head(self, vision_x, gaze_x):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, n1, D)

        """

        vision_x = self.norm_media(vision_x)
        q = self.norm_gaze(gaze_x)
        h = self.heads
        


        q = self.to_q(gaze_x)
        kv_input = vision_x
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        #getting rid of multi-head attention- not required and makes things complex 
        q, k, v = rearrange_many((q, k, v), "b t n (h d) -> b h t n d", h=h)
        q = q * self.scale
        # print("shape of query is ",q.shape)
        # print("shape of key and value is ",k.shape)

        # attention
        sim = einsum("... i d, ... j d  -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        #print("shape of attention weights is ",attn.shape)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h t n d -> b t n (h d)", h=h)
        attn = rearrange(attn, "b h t n d -> b t n (h d)", h=h)
        return self.to_attention(attn),self.to_out(out)



#to-do: change this so that output is 256x1, not 16x16
def calculate_gaze_proportions(image, num_patches_vertical, num_patches_horizontal):
    # Assume image is a grayscale image where the intensity represents the gaze focus
    total_gaze = np.sum(image)  # Total sum of pixel values in the image

    # Dimensions for each patch
    patch_height = image.shape[0] // num_patches_vertical
    patch_width = image.shape[1] // num_patches_horizontal

    # Initialize an array to hold the gaze proportion for each patch
    gaze_proportions = np.zeros((num_patches_vertical, num_patches_horizontal))

    for i in range(num_patches_vertical):
        for j in range(num_patches_horizontal):
            # Extract the patch
            patch = image[i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width]
            # Sum the pixel values in the patch
            patch_sum = np.sum(patch)
            # Calculate the proportion of total gaze this patch contains
            gaze_proportions[i, j] = patch_sum / total_gaze if total_gaze != 0 else 0

    return gaze_proportions






class GazeVisionAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_gaze = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)  # Gaze features as queries
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)  # Vision features as keys and values
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, vision_features, gaze_features):
        # Normalization
        # print("Shape of input (vision_features):", vision_features.shape)
        vision_features = self.norm_media(vision_features)
        gaze_features = self.norm_gaze(gaze_features)
        kl_div_loss = 0

        # Prepare queries, keys, and values
        q = self.to_q(gaze_features)
        kv = self.to_kv(vision_features)
        k, v = kv.chunk(2, dim=-1)
        # print("Shape of input (q):", q.shape)
        # print("Shape of target (v):", v.shape)



        # Head rearrangement for multi-head attention
        q, k, v = [rearrange(x, 'b t f v (h d) -> (b t f) h v d', h=self.heads) for x in [q, k, v]]
        q = q * self.scale

        # Attention calculation
        dots = torch.einsum('bhid,bhjd->bhij', q, k)
        attn = F.softmax(dots, dim=-1)  # This is Q

        # Weighted sum of values per the attention probabilities
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, '(b t f) h v d -> b t f v (h d)', b=vision_features.shape[0], t=vision_features.shape[1], f=vision_features.shape[2], h=self.heads)

        P = F.softmax(rearrange(gaze_features, 'b t f v (h d) -> (b t f) h v d', h=self.heads), dim=-1)
        # Assuming P needs to be reduced
        # Correcting P to match attn's last dimension of 256
        if P.shape[-1] == 128 and attn.shape[-1] == 256:
             P = P.repeat_interleave(2, dim=-1)  # Repeating elements to expand dimensions

        # print("Shape of input (attn):", attn.shape)
        # print("Shape of target (P):", P.shape)

        # Compute KL divergence
        kl_div_loss = F.kl_div(attn.log(), P, reduction='batchmean')

        # Final output projection
        final_output = self.to_out(out)
        return final_output, kl_div_loss




# class GazeVisionAttention(nn.Module):
#     def __init__(self, *, dim, dim_head=64, heads=8):
#         super().__init__()
#         self.scale = dim_head ** -0.5
#         self.heads = heads
#         inner_dim = dim_head * heads

#         self.norm_media = nn.LayerNorm(dim)
#         self.norm_gaze = nn.LayerNorm(dim)

#         self.to_q = nn.Linear(dim, inner_dim, bias=False)  # Gaze features as queries
#         self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)  # Vision features as keys and values
#         self.to_out = nn.Linear(inner_dim, dim, bias=False)

#     def forward(self, vision_features, gaze_features):
#         """
#         Args:
#             vision_features (torch.Tensor): Vision features, used as keys and values
#                 shape (b, T, F, V, D)
#             gaze_features (torch.Tensor): Gaze features, used as queries
#                 shape (b, T, F, V, D)
#         """
#         # Normalization
#         vision_features = self.norm_media(vision_features)
#         gaze_features = self.norm_gaze(gaze_features)

#         # Prepare queries, keys, and values
#         q = self.to_q(gaze_features)
#         kv = self.to_kv(vision_features)  # This combines keys and values generation
#         k, v = kv.chunk(2, dim=-1)
        
#         # Head rearrangement for multi-head attention
#         q, k, v = [rearrange(x, 'b t f v (h d) -> (b t f) h v d', h=self.heads) for x in [q, k, v]]
#         q = q * self.scale

#         # Attention calculation
#         dots = torch.einsum('bhid,bhjd->bhij', q, k)  # Batch matrix multiplication
#         attn = torch.nn.functional.softmax(dots, dim=-1)  # Softmax over last dim to create probabilities

#         # Weighted sum of values per the attention probabilities
#         out = torch.einsum('bhij,bhjd->bhid', attn, v)
#         out = rearrange(out, '(b t f) h v d -> b t f v (h d)', b=vision_features.shape[0], t=vision_features.shape[1], f=vision_features.shape[2], h=self.heads)

#         # Final output projection
#         return self.to_out(out)







class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, n1, D)
            latent (torch.Tensor): latent features
                shape (b, T, n2, D)
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        h = self.heads

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), "b t n (h d) -> b h t n d", h=h)
        q = q * self.scale

        # attention
        sim = einsum("... i d, ... j d  -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h t n d -> b t n (h d)", h=h)
        return self.to_out(out)


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=6,
        dim_head=64,
        heads=8,
        num_latents=64,
        max_num_media=None,
        max_num_frames=None,
        ff_mult=4,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.frame_embs = (
            nn.Parameter(torch.randn(max_num_frames, dim))
            if exists(max_num_frames)
            else None
        )
        self.media_time_embs = (
            nn.Parameter(torch.randn(max_num_media, 1, dim))
            if exists(max_num_media)
            else None
        )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, F, v, D)
        Returns:
            shape (b, T, n, D) where n is self.num_latents
        """
        b, T, F, v = x.shape[:4]
        # frame and media time embeddings
        if exists(self.frame_embs):
            frame_embs = repeat(self.frame_embs[:F], "F d -> b T F v d", b=b, T=T, v=v)
            x = x + frame_embs
            #print("Shape of frame embedding is ", frame_embs.shape)
        # x = rearrange(
        #     x, "b T F v d -> b T (F v) d"
        # )  # flatten the frame and spatial dimensions

        if exists(self.media_time_embs):
            #print("Shape of tensor time embs is ", self.media_time_embs.shape)
            #print("Shape of x tensor is ", x.shape)
            x = x + self.media_time_embs[:T]

        # blocks
        # print("Shape of latents before is ", self.latents.shape)
        latents = repeat(self.latents, "n d -> b T n d", b=b, T=T) #basically one query nd which is repeated 
        # print("Shape of latents after is ", latents.shape)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        return self.norm(latents)


# gated cross attention
class MaskedCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_visual,
        dim_head=64,
        heads=8,
        only_attend_immediate_media=True,
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_visual, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # whether for text to only attend to immediate preceding image, or all previous images
        self.only_attend_immediate_media = only_attend_immediate_media

    def forward(self, x, media, media_locations=None, use_cached_media=False):
        """
        Args:
            x (torch.Tensor): text features
                shape (B, T_txt, D_txt)
            media (torch.Tensor): image features
                shape (B, T_img, n, D_img) where n is the dim of the latents
            media_locations: boolean mask identifying the media tokens in x
                shape (B, T_txt)
            use_cached_media: bool
                If true, treat all of x as if they occur after the last media
                registered in media_locations. T_txt does not need to exactly
                equal media_locations.shape[1] in this case
        """

        if not use_cached_media:
            assert (
                media_locations.shape[1] == x.shape[1]
            ), f"media_location.shape is {media_locations.shape} but x.shape is {x.shape}"

        T_txt = x.shape[1]
        _, T_img, n = media.shape[:3]
        h = self.heads

        x = self.norm(x)

        q = self.to_q(x)
        media = rearrange(media, "b t n d -> b (t n) d")

        k, v = self.to_kv(media).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=h)

        q = q * self.scale

        sim = einsum("... i d, ... j d -> ... i j", q, k)

        if exists(media_locations):
            media_time = torch.arange(T_img, device=x.device) + 1

            if use_cached_media:
                # text time is set to the last cached media location
                text_time = repeat(
                    torch.count_nonzero(media_locations, dim=1),
                    "b -> b i",
                    i=T_txt,
                )
            else:
                # at each boolean of True, increment the time counter (relative to media time)
                text_time = media_locations.cumsum(dim=-1)

            # text time must equal media time if only attending to most immediate image
            # otherwise, as long as text time is greater than media time (if attending to all previous images / media)
            mask_op = torch.eq if self.only_attend_immediate_media else torch.ge

            text_to_media_mask = mask_op(
                rearrange(text_time, "b i -> b 1 i 1"),
                repeat(media_time, "j -> 1 1 1 (j n)", n=n),
            )
            sim = sim.masked_fill(~text_to_media_mask, -torch.finfo(sim.dtype).max)

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        if exists(media_locations) and self.only_attend_immediate_media:
            # any text without a preceding media needs to have attention zeroed out
            text_without_media_mask = text_time == 0
            text_without_media_mask = rearrange(
                text_without_media_mask, "b i -> b 1 i 1"
            )
            attn = attn.masked_fill(text_without_media_mask, 0.0)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class GatedCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_visual,
        dim_head=64,
        heads=8,
        ff_mult=4,
        only_attend_immediate_media=True,
    ):
        super().__init__()
        self.attn = MaskedCrossAttention(
            dim=dim,
            dim_visual=dim_visual,
            dim_head=dim_head,
            heads=heads,
            only_attend_immediate_media=only_attend_immediate_media,
        )
        self.attn_gate = nn.Parameter(torch.tensor([0.0]))

        self.ff = FeedForward(dim, mult=ff_mult)
        self.ff_gate = nn.Parameter(torch.tensor([0.0]))

    def forward(
        self,
        x,
        media,
        media_locations=None,
        use_cached_media=False,
    ):
        x = (
            self.attn(
                x,
                media,
                media_locations=media_locations,
                use_cached_media=use_cached_media,
            )
            * self.attn_gate.tanh()
            + x
        )
        x = self.ff(x) * self.ff_gate.tanh() + x

        return x
