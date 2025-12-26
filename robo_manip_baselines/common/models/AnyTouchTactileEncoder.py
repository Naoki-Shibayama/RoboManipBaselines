import copy
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.ops.misc import FrozenBatchNorm2d
from transformers import CLIPConfig, CLIPVisionConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import (
    CLIP_VISION_INPUTS_DOCSTRING,
    CLIPVisionTransformer,
)
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

from robo_manip_baselines.common import DataKey


def load_model_from_multi_clip(ckpt, model, use_same_patchemb=False):
    new_ckpt = {}
    for key, item in ckpt.items():
        if (
            "touch_model" in key
            or "touch_projection" in key
            or "sensor_token" in key
            and "sensor_token_proj" not in key
        ):
            new_ckpt[key.replace("touch_mae_model.", "")] = copy.deepcopy(item)
        if use_same_patchemb:
            if "video_patch_embedding" in key:
                new_key = key.replace("touch_mae_model.", "")
                new_ckpt[
                    new_key.replace(
                        "video_patch_embedding",
                        "touch_model.embeddings.patch_embedding",
                    )
                ] = copy.deepcopy(item)

    for k, v in model.named_parameters():
        if k not in new_ckpt.keys():
            new_ckpt[k] = v

    model.load_state_dict(new_ckpt, strict=True)

    return model


class AnyTouch(nn.Module):
    def __init__(
        self,
        model_config: CLIPConfig,
        num_frames,
        add_time_attn,
        tube_size,
        use_sensor_token,
        use_same_patchemb,
        pooling,
        n_obs_steps,
    ):
        super(AnyTouch, self).__init__()

        model_config.vision_config.num_frames = num_frames
        model_config.vision_config.tube_size = tube_size

        self.use_sensor_token = use_sensor_token
        self.use_same_patchemb = use_same_patchemb
        self.n_obs_steps = n_obs_steps

        if self.use_sensor_token:
            self.sensor_token = nn.Parameter(
                torch.zeros(10, 5, model_config.vision_config.hidden_size)
            )

        self.touch_model = CLIPVisionTransformer(model_config.vision_config)
        self.touch_projection = nn.Linear(
            model_config.vision_config.hidden_size,
            model_config.projection_dim,
            bias=False,
        )

        if self.use_same_patchemb:
            self.touch_model.embeddings.patch_embedding = nn.Conv3d(
                in_channels=model_config.vision_config.num_channels,
                out_channels=self.touch_model.embeddings.embed_dim,
                kernel_size=(
                    3,
                    self.touch_model.embeddings.patch_size,
                    self.touch_model.embeddings.patch_size,
                ),
                stride=(
                    3,
                    self.touch_model.embeddings.patch_size,
                    self.touch_model.embeddings.patch_size,
                ),
                bias=False,
            )

        self.pooling = pooling

        self.touch_model.forward = self.touch_forward
        self.touch_model.embeddings.forward = self.emb_forward

    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=BaseModelOutputWithPooling, config_class=CLIPVisionConfig
    )
    def touch_forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        sensor_type=None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """

        # a = self.sensor_token[sensor_type]
        # print(a.shape)
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.touch_model.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.touch_model.config.output_hidden_states
        )
        return_dict = (
            return_dict
            if return_dict is not None
            else self.touch_model.config.use_return_dict
        )

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.touch_model.embeddings(
            pixel_values, sensor_type=sensor_type
        )
        hidden_states = self.touch_model.pre_layrnorm(hidden_states)

        encoder_outputs = self.touch_model.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.touch_model.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def emb_forward(
        self, pixel_values: torch.FloatTensor, noise=None, sensor_type=None
    ) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.touch_model.embeddings.patch_embedding.weight.dtype
        patch_embeds = self.touch_model.embeddings.patch_embedding(
            pixel_values.to(dtype=target_dtype)
        )  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        pos_emb = self.touch_model.embeddings.position_embedding(
            self.touch_model.embeddings.position_ids
        )

        embeddings = patch_embeds + pos_emb[:, 1:, :]

        class_embeds = self.touch_model.embeddings.class_embedding + pos_emb[:, 0, :]
        class_embeds = class_embeds.expand(batch_size, 1, -1)

        if self.use_sensor_token:
            sensor_emb = self.sensor_token[sensor_type.repeat(self.n_obs_steps)]
            embeddings = torch.cat([class_embeds, sensor_emb, embeddings], dim=1)
        else:
            embeddings = torch.cat([class_embeds, embeddings], dim=1)
        # embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings

    def forward(self, x, sensor_type=None):
        if self.use_same_patchemb:
            x = x.unsqueeze(1).repeat(1, 3, 1, 1, 1)

        # print(x.shape, sensor_type.shape)

        with torch.no_grad():
            x = self.touch_model(x, sensor_type=sensor_type)
            if self.pooling == "cls":
                out = self.touch_projection(x.pooler_output)
            else:
                out = self.touch_projection(x.last_hidden_state)

        if self.pooling == "global":
            if self.use_sensor_token:
                out = out[:, 6:, :].mean(dim=1)
            else:
                out = out[:, 1:, :].mean(dim=1)

        return out


class AnyTouchTactileEncoder(nn.Module):
    def __init__(
        self,
        shape_meta_obs: dict,
        anytouch_checkpoint_path: str,
        anytouch_config: CLIPConfig,
        n_obs_steps: int = 1,
        output_dim: int = 64,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        tactile_pooling: str = "global",
        use_sensor_token: bool = False,
        use_same_patchemb: bool = False,
    ):
        super().__init__()
        self.shape_meta_obs = shape_meta_obs
        self.image_rgb_keys = [
            k for k, v in shape_meta_obs["image"].items() if v.get("type") == "rgb"
        ]
        self.tactile_rgb_keys = [
            k
            for k, v in shape_meta_obs["tactile_image"].items()
            if v.get("type") == "rgb"
        ]

        # ==== (1) CNN encoder (ResNet) ====
        weights = ResNet18_Weights.IMAGENET1K_V1
        resnet = resnet18(weights=weights, norm_layer=FrozenBatchNorm2d)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])  # remove avgpool/fc
        resnet_out_dim = 512

        # ==== (2) Learnable positional embedding ====
        _, self.image_height, self.image_width = shape_meta_obs["image"][
            self.image_rgb_keys[0]
        ]["shape"]
        num_patches = round(self.image_width / 32) * round(self.image_height / 32)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, resnet_out_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ==== (3) Transformer encoder ====
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=resnet_out_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # ==== (4) AnyTouch tactile encoder ====
        self.tactile_encoder = AnyTouch(
            anytouch_config,
            1,
            False,
            1,
            use_sensor_token,
            use_same_patchemb,
            tactile_pooling,
            n_obs_steps,
        )
        loaded_checkpoint = torch.load(anytouch_checkpoint_path, map_location="cpu")[
            "model"
        ]
        self.tactile_encoder = load_model_from_multi_clip(
            loaded_checkpoint, self.tactile_encoder, use_same_patchemb
        )
        self.tactile_projection = nn.Linear(
            anytouch_config.projection_dim, num_patches * resnet_out_dim
        )

        # ==== (5) State encoder ====
        state_dim = shape_meta_obs["state"]["shape"][0]
        state_feature_dim = 32
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, state_feature_dim),
            nn.ReLU(),
        )

        # ==== (6) Output head ====
        concat_dim = (
            num_patches
            * resnet_out_dim
            * (len(self.image_rgb_keys) + len(self.tactile_rgb_keys))
            + state_feature_dim
        )
        self.output_mlp = nn.Sequential(
            nn.Linear(concat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(
        self,
        obs: dict,
    ):
        """
        Args:
            obs["state"]: [B, state_dim]
            obs["image"]["<camera_name>_rgb_image"]: [B, 3, image_height, image_width]
            obs["tactile_image"]["<tactile_camera_name>_rgb_image"]: [B, 3, image_height, image_width]
            obs["tactile_sensor_type"]: [B, 1]
        Returns:
            output: [B, output_dim] (for Diffusion Policy conditioning)
        """
        B = obs["state"].shape[0]
        image_feature_all = []
        tactile_feature_all = []

        for k in self.image_rgb_keys:
            image = obs["image"][k]  # [B, 3, H, W]
            x = self.cnn(image)  # [B, resnet_out_dim, H/32, W/32]
            x = x.flatten(2).transpose(1, 2)  # [B, num_patches, resnet_out_dim]
            x = x + self.pos_embed  # add positional embedding

            x = self.transformer_encoder(x)  # [B, num_patches, resnet_out_dim]

            image_feature_all.append(x.reshape(B, -1))

        image_feature_all = torch.cat(image_feature_all, dim=-1)

        # ==== state encoding ====
        state_feature = self.state_encoder(obs["state"])  # [B, state_feature_dim]

        # ==== tactile encoding ====
        for k in self.tactile_rgb_keys:
            image = obs["tactile_image"][k]
            sensor_type = obs["tactile_sensor_type"][DataKey.get_camera_name(k)]
            tact = self.tactile_encoder(image, sensor_type=sensor_type).reshape(B, -1)
            tact = self.tactile_projection(tact)
            tactile_feature_all.append(tact)

        tactile_feature_all = torch.cat(tactile_feature_all, dim=-1)

        # ==== fusion and output ====
        feature_all = torch.cat(
            [image_feature_all, tactile_feature_all, state_feature], dim=-1
        )
        output = self.output_mlp(feature_all)  # [B, output_dim]

        return output
