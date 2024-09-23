import torch
import torch.nn as nn

from .adapter import ResMLP
from typing import List

_LAYER_PREFIX = "layer_"

def from_sample_of_embeddings(embeddings, population_size=None):
  """Initialize by drawing vectors from the embedding table.
    
  Note:
    If not provided, the population size used is the full possibility of the
    vector space.
 
  Args:
    embeddings: [V, H] The embeddings to draw vectors from. can be extract 
      by model_seq.esm2.embed_tokens.weight
    population_size: Limit the drawing to the first `population_size` vectors.
  
  Returns:
    A closure over the embedding table that can be used as a flax initializer.
  """
  if population_size is None:
    population_size = embeddings.shape[0]
  if population_size <= 0:
    raise ValueError(f"Cannot sample from a population less than zero. Got "
                     f"{population_size}")
  if population_size > embeddings.shape[0]:
    # logging.warning(
    #    "The requested `population_size` (%d) is larger than the "
    #    "total available embeddings (%d). Setting "
    #    "`population_size` to the embedding size.", population_size,
    #    embeddings.shape[0])
    print("The requested `population_size` (%d) is larger than the "
          "total available embeddings (%d). Setting "
          "`population_size` to the embedding size.", population_size,
          embeddings.shape[0])

    population_size = embeddings.shape[0]

  # Because our sampling is done with jax (so that we can make use of the rng
  # key), we need our embeddings to be in jax, otherwise we get errors because
  # the indices will be a jax tracer and it fails when it is converted to numpy
  # to lookup values in a number array. This call pins the embeddings to cpu so
  # we don't waste TPU memory carrying it around.
  embeddings = embeddings.cpu()

  def initialize_from_embedding_sample(shape, rng=1234):
    """Sample from the embedding table, without replacement.
    
    Note:
      If the number of prompt tokens requested is larger than the total number
      of vectors we are drawing from (`population_size`) we do sampling with
      replacement.
    
    Args:
      rng: The rng seed used in our sampling.
      shape: The shape of the prompt variable. shape[0] tells us how many
        vectors to sample.
    
    Raises:
      ValueError if the number of features in the embedding table do not match
      the number of features in the prompt.
    
    Returns:
      A sample of the embedding table as a jax array. [P, H]
    """
    if embeddings.shape[-1] != shape[-1]:
      raise ValueError(
          "Shape mismatch between the number of features in the "
          f"embeddings: {embeddings.shape[-1]} and the requested prompt shape "
          f"{shape[-1]}.")
    replace = False
    if shape[0] > population_size:
      print("Prompt Length: %d is larger than the number of vectors "
            "to draw from: %d. Switching to draws with replacement.", shape[0],
            population_size)
      replace = True

    # set the seed for torch random number generator
    torch.manual_seed(rng)
    if replace:
      index = torch.randint(population_size, size=(shape[0],))
    else:
      index = torch.multinomial(torch.ones(
          population_size), shape[0], replacement=False)

    return embeddings[index].clone().detach()

  return initialize_from_embedding_sample




class PrefixTuning(nn.Module):
  """A module that produces a learnable prompt.

  Args:
    backbone (nn.Module): The backbone model.
    prompt_len (int): The length of the prompt.
    input_seq_len (int, optional): The length of the input sequence.
      Defaults to 70.
    prompt_layer_indices (int, optional): The index of the layer in the backbone
      where the prompt is added. Defaults to 0.
    mlp_bottleneck_size (int, optional): The size of the MLP bottleneck layer.
      Defaults to 0.
    device (str, optional): The device to run the module on. Defaults to "cpu".
  """

  def __init__(self,
              backbone: nn.Module,
              prompt_len: int = None,
              input_seq_len: int = 70,
              prompt_layer_indices: List[int] = [0],
              device="cpu"):
    super(PrefixTuning, self).__init__()
    
    assert isinstance(prompt_layer_indices, list), ("prompt_layer_indices "
                                                    "must be a list")
    
    self.prompt_len = prompt_len
    self.input_seq_len = input_seq_len
    self.prompt_layer_indices = prompt_layer_indices
    self.device = device

    token_embed_table = backbone.embed_tokens.weight
    embed_size = token_embed_table.shape[-1]

    num_backbone_layers = len(backbone.layers)
    
    if max(self.prompt_layer_indices) > num_backbone_layers - 1:
      raise ValueError(f"prompt layer index {max(self.prompt_layer_indices)} "
                       f"is out of range. The number of layers in "
                       f"the backbone is {num_backbone_layers}")
    
    if prompt_len is not None:
      self.prompt_layer_dict = nn.ParameterDict()
      for idx in self.prompt_layer_indices:
        self.prompt_layer_dict[f"{_LAYER_PREFIX}{idx}"] = (
          from_sample_of_embeddings(token_embed_table)([prompt_len, 
                                                        embed_size]))


  def prefix_concat(self, prompt_weight, input_embedding) -> torch.Tensor:
    """Concatenate prompt_weight to the beginning of input_embed.

    Args:
      prompt_weight: [B, P, H] The prompt weight.
      input_embed: [B, T, H] The embedded input.

    Returns:
      The input with the prompt concatenated to the front. [B, P + T, H]
    """
    return torch.cat((prompt_weight, input_embedding), dim=1)

  def expand_to_batch(self, x, y):
    """
    Expands the input tensor x to match the batch size of the target tensor y.

    Args:
      x (torch.Tensor): The input tensor of shape (input_size).
      y (torch.Tensor): The target tensor of shape (batch_size, target_size).

    Returns:
      torch.Tensor: The expanded input tensor of shape (batch_size, input_size).
    """
    batch_size = y.shape[0]
    expanded_x = x.unsqueeze(0)
    tiled_x = expanded_x.expand(batch_size, -1, -1)
    return tiled_x

  def forward(self,
              input_embedding: torch.Tensor,
              layer_idx: int = 0) -> torch.Tensor:
    """
    Forward pass of the model adding prompt weight to a specific embedding.

    Args:
      input_embedding (torch.Tensor): Input embedding tensor of shape [B, T, E].
      layer_idx (int): Index of the layer.

    Returns:
      torch.Tensor: Output embedding tensor of shape [B, T, E].
    """

    if (self.prompt_len == 0) or (layer_idx not in self.prompt_layer_indices):
      return input_embedding

    prompted_embedding = self.expand_to_batch(
      self.prompt_layer_dict[f"{_LAYER_PREFIX}{layer_idx}"],
      input_embedding)

    # [B,PromptT,E] + [B,T,E] => [B,PromptT+T,E]
    input_embedding = self.prefix_concat(prompted_embedding, input_embedding)
    return input_embedding
  