import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from typing import List, Dict, Optional, Tuple, Callable
from efficientnet_pytorch import EfficientNet
from vint_train.models.vint.self_attention import PositionalEncoding

class NoMaD_ViNT(nn.Module):
    def __init__(
        self,
        context_size: int = 5, # Number of past observations to use as context
        obs_encoder: Optional[str] = "efficientnet-b0",
        obs_encoding_size: Optional[int] = 512,
        mha_num_attention_heads: Optional[int] = 2, # Number of attention heads in Multi-Head Attention
        mha_num_attention_layers: Optional[int] = 2, # Number of layers in Multi-Head Attention
        mha_ff_dim_factor: Optional[int] = 4, # Factor for feed-forward dimension in MHA
    ) -> None: # Return type hint: None
        """
        NoMaD ViNT Encoder class

        High-level model architecture:
        - Observation Encoder: Takes in the most recent observation image and goal image.
        - Goal        Encoder:
        - Compression Layer  : Optionally unifies encoder dimension with expected encoding feature dimension.
        - Transformer Module :
        """
        super().__init__()
        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = obs_encoding_size
        self.context_size = context_size

        # Initialize the observation encoder
        if obs_encoder.split("-")[0] == "efficientnet":
            self.obs_encoder = EfficientNet.from_name(obs_encoder, in_channels=3) # load pre-trained for RGB images
            self.obs_encoder = replace_bn_with_gn(self.obs_encoder) # BatchNorm -> GroupNorm
            self.num_obs_features = self.obs_encoder._fc.in_features # Get the number of features from the EfficientNet's fully connected layer
            self.obs_encoder_type = "efficientnet" # Set the observation encoder type
        else:
            raise NotImplementedError

        # Initialize the goal encoder
        self.goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=6) # Load EfficientNet-B0 for 6-channel images (obs+goal)
        self.goal_encoder = replace_bn_with_gn(self.goal_encoder) # Replace Batch Normalization with Group Normalization
        self.num_goal_features = self.goal_encoder._fc.in_features # Get the number of features from the goal encoder's FC layer

        # Initialize compression layers if necessary (for dimensionality matching)
        if self.num_obs_features != self.obs_encoding_size: # If feature size doesn't match desired encoding size
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.obs_encoding_size) # Add a linear layer to compress observation features
        else:
            self.compress_obs_enc = nn.Identity() # Otherwise, use an identity layer (no change)

        if self.num_goal_features != self.goal_encoding_size: # If goal feature size doesn't match desired encoding size
            self.compress_goal_enc = nn.Linear(self.num_goal_features, self.goal_encoding_size) # Add a linear layer to compress goal features
        else:
            self.compress_goal_enc = nn.Identity() # Otherwise, use an identity layer

        # Initialize positional encoding and self-attention layers
        self.positional_encoding = PositionalEncoding(self.obs_encoding_size, max_seq_len=self.context_size + 2) # Initialize positional encoding for sequence data (context + current_obs + goal) # to represent time information
        self.sa_layer = nn.TransformerEncoderLayer( # Define a single Transformer Encoder layer
            d_model=self.obs_encoding_size, # Dimensionality of the input features
            nhead=mha_num_attention_heads, # Number of attention heads
            dim_feedforward=mha_ff_dim_factor*self.obs_encoding_size, # Dimensionality of the feed-forward network
            activation="gelu", # Activation function (GELU)
            batch_first=True, # Expect batch dimension first (batch, seq, feature)
            norm_first=True # Apply layer normalization before other sublayers
        )
        self.sa_encoder = nn.TransformerEncoder(self.sa_layer, num_layers=mha_num_attention_layers) # Stack multiple Transformer Encoder layers

        # Definition of the goal mask (convention: 0 = no mask, 1 = mask)
        self.goal_mask = torch.zeros((1, self.context_size + 2), dtype=torch.bool) # Create a mask tensor initialized to false (no mask)
        self.goal_mask[:, -1] = True # Mask out the goal token (last token in the sequence)
        self.no_mask = torch.zeros((1, self.context_size + 2), dtype=torch.bool) # Create a tensor representing no mask
        self.all_masks = torch.cat([self.no_mask, self.goal_mask], dim=0) # Concatenate no_mask and goal_mask for easy selection
        self.avg_pool_mask = torch.cat([1 - self.no_mask.float(), (1 - self.goal_mask.float()) * ((self.context_size + 2)/(self.context_size + 1))], dim=0) # Create masks for weighted average pooling


    def forward(self, obs_img: torch.tensor, goal_img: torch.tensor, input_goal_mask: torch.tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # obs_img: tensor of observation images (batch_size, context_size*3, H, W)
        # goal_img: tensor of goal images (batch_size, 3, H, W)
        # input_goal_mask: optional tensor indicating whether to mask the goal

        device = obs_img.device # Get the device (CPU/GPU) of the input tensor

        # Initialize the goal encoding
        goal_encoding = torch.zeros((obs_img.size()[0], 1, self.goal_encoding_size)).to(device) # Initialize goal encoding tensor with zeros

        # Get the input goal mask
        if input_goal_mask is not None: # If a goal mask is provided
            goal_mask = input_goal_mask.to(device) # Move the mask to the correct device

        # Get the goal encoding
        obsgoal_img = torch.cat([obs_img[:, 3*self.context_size:, :, :], goal_img], dim=1) # Concatenate the most recent observation image and the goal image
        obsgoal_encoding = self.goal_encoder.extract_features(obsgoal_img) # Extract features using the goal encoder
        obsgoal_encoding = self.goal_encoder._avg_pooling(obsgoal_encoding) # Apply average pooling  # NOTE: What dimension does this reduce (if any)?

        if self.goal_encoder._global_params.include_top: # If the EfficientNet model includes the top classification layer
            obsgoal_encoding = obsgoal_encoding.flatten(start_dim=1) # Flatten the features
            obsgoal_encoding = self.goal_encoder._dropout(obsgoal_encoding) # Apply dropout
        obsgoal_encoding = self.compress_goal_enc(obsgoal_encoding) # Compress the goal encoding to the desired size

        if len(obsgoal_encoding.shape) == 2: # If encoding is 2D (batch_size, features)
            obsgoal_encoding = obsgoal_encoding.unsqueeze(1) # Add sequence dimension (batch_size, 1, features)
        assert obsgoal_encoding.shape[2] == self.goal_encoding_size
        goal_encoding = obsgoal_encoding # NOTE Important: assign processed encoding to goal_encoding

        # Get the observation encoding
        obs_img = torch.split(obs_img, 3, dim=1) # Split the observation images tensor into individual images (list of tensors)
        obs_img = torch.concat(obs_img, dim=0) # Concatenate the list of image tensors along the batch dimension

        obs_encoding = self.obs_encoder.extract_features(obs_img)
        obs_encoding = self.obs_encoder._avg_pooling(obs_encoding)
        if self.obs_encoder._global_params.include_top: # if encoder includes the top classification layer
            obs_encoding = obs_encoding.flatten(start_dim=1)
            obs_encoding = self.obs_encoder._dropout(obs_encoding) # Apply dropout
        obs_encoding = self.compress_obs_enc(obs_encoding) # Match dimensionality
        obs_encoding = obs_encoding.unsqueeze(1) # Add a sequence dimension
        obs_encoding = obs_encoding.reshape((self.context_size+1, -1, self.obs_encoding_size)) # (context_size+1, batch_size, encoding_size)
        obs_encoding = torch.transpose(obs_encoding, 0, 1) # (batch_size, context_size+1, encoding_size)
        obs_encoding = torch.cat((obs_encoding, goal_encoding), dim=1)  # NOTE: important; obs & goal encodings are fused

        # If a goal mask is provided, mask some of the goal tokens
        if goal_mask is not None:
            no_goal_mask = goal_mask.long() # Convert boolean mask to long for indexing
            src_key_padding_mask = torch.index_select(self.all_masks.to(device), 0, no_goal_mask) # Select the appropriate mask (goal_mask or no_mask)
        else:
            src_key_padding_mask = None

        # Apply positional encoding
        if self.positional_encoding:
            obs_encoding = self.positional_encoding(obs_encoding)

        obs_encoding_tokens = self.sa_encoder(obs_encoding, src_key_padding_mask=src_key_padding_mask) # Pass encodings through Transformer
        if src_key_padding_mask is not None:
            avg_mask = torch.index_select(self.avg_pool_mask.to(device), 0, no_goal_mask).unsqueeze(-1) # Select the appropriate average pooling mask
            obs_encoding_tokens = obs_encoding_tokens * avg_mask # Apply the mask for weighted averaging
        obs_encoding_tokens = torch.mean(obs_encoding_tokens, dim=1) # Average the tokens across the sequence dimension

        return obs_encoding_tokens # final representation



# Utils for Group Norm
def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module
