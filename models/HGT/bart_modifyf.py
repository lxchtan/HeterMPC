import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from transformers.modeling_bart import (
    BartEncoder,
    invert_mask,
    BaseModelOutput
)

from .HGT import HGTLayer
# from dataloaders import MAX_UTTERANCE_NUM, MAX_SPEAKER_NUM


class HGTModel(nn.Module):
  def __init__(self, config, args, node_dict, edge_dict):
    super().__init__()
    self.node_dict = node_dict
    self.edge_dict = edge_dict
    self.n_inp = self.n_hid = self.n_out = config.hidden_size
    self.n_layers = args.num_gat_layers
    self.adapt_ws = nn.ModuleList([nn.Linear(self.n_inp, self.n_hid) for _ in range(len(node_dict))])
    self.gcs = nn.ModuleList([HGTLayer(self.n_hid, self.n_hid, node_dict, edge_dict, args.num_gat_heads, use_norm=True) for _ in range(self.n_layers)])
    self.out = nn.ModuleList([nn.Linear(self.n_hid, self.n_out) for _ in range(len(node_dict))])

  def forward(self, G, feats):
    h = {}
    for ntype in G.ntypes:
      n_id = self.node_dict[ntype]
      h[ntype] = F.gelu(self.adapt_ws[n_id](feats[ntype]))
    for i in range(self.n_layers):
      h = self.gcs[i](G, h)

    out = {}
    for ntype in G.ntypes:
      n_id = self.node_dict[ntype]
      out[ntype] = self.out[n_id](h[ntype])
    return out


class TXHEncoder(BartEncoder):
  """
  Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
  :class:`EncoderLayer`.

  Args:
      config: BartConfig
  """

  def __init__(self, config, embed_tokens, args):
    super().__init__(config, embed_tokens)

    self.config = config
    self.MAX_SPEAKER_NUM = args.MAX_SPEAKER_NUM
    self.MAX_UTTERANCE_NUM = args.MAX_UTTERANCE_NUM
    node_dict = {"utterance": 0, "speaker": 1}
    edge_dict = {'from': 0, 'to': 1, 'speak': 2, 'speaked': 3, 'aim_to': 4, 'get_from': 5, 'non_from': 6, 'non_to': 7, 'non_aim_to': 8, 'non_get_from': 9}
    self.node_dict = node_dict
    self.edge_dict = edge_dict

    self.gnn_layer_start = args.gnn_layer_start
    self.ignore_last_layer = getattr(args, "ignore_last_layer", False)
    self.num_layers = config.num_hidden_layers
    self.gat_layer = nn.ModuleList([HGTModel(config, args, node_dict, edge_dict) for _ in range(self.num_layers - self.gnn_layer_start)])
    self.combine_layer = nn.ModuleList([nn.ModuleList([
        nn.Linear(config.hidden_size * 2, config.hidden_size)
        for _ in range(len(node_dict))
    ]) for _ in range(self.num_layers - self.gnn_layer_start)])

    self.speakers = nn.Embedding(self.MAX_SPEAKER_NUM, config.hidden_size)

  def embedding_ids(self, input_ids):
    inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
    embed_pos = self.embed_positions(input_ids)
    x = inputs_embeds + embed_pos
    x = self.layernorm_embedding(x)
    x = F.dropout(x, p=self.dropout, training=self.training)
    return x

  def forward(
      self, input_ids, graph, attention_mask=None, output_attentions=False, output_hidden_states=False, return_dict=False
  ):
    """
    Args:
        input_ids (LongTensor): tokens in the source language of shape
            `(batch, src_len)`
        attention_mask (torch.LongTensor): indicating which indices are padding tokens

    Returns:
        BaseModelOutput or Tuple comprised of:

            - **x** (Tensor): the last encoder layer's output of shape `(src_len, batch, embed_dim)`
            - **encoder_states** (tuple(torch.FloatTensor)): all intermediate hidden states of shape `(src_len,
              batch, embed_dim)`. Only populated if *output_hidden_states:* is True.
            - **all_attentions** (tuple(torch.FloatTensor)): Attention weights for each layer.
            During training might not be of length n_layers because of layer dropout.
    """
    # check attention mask and invert
    if attention_mask is not None:
      attention_mask = invert_mask(attention_mask)

    inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
    embed_pos = self.embed_positions(input_ids)
    x = inputs_embeds + embed_pos
    x = self.layernorm_embedding(x)
    x = F.dropout(x, p=self.dropout, training=self.training)

    batch_size = x.size(0)//self.MAX_UTTERANCE_NUM
    speaker_hidden = self.speakers(torch.arange(0, self.MAX_SPEAKER_NUM, dtype=torch.long, device=x.device
                                                ).view(1, -1).repeat(batch_size, 1)).view(-1, self.config.hidden_size)  # (bz*MAX_SPEAKER_NUM, hidden_dim)


    # B x T x C -> T x B x C
    x = x.transpose(0, 1)

    encoder_states = [] if output_hidden_states else None
    all_attentions = () if output_attentions else None
    for i, encoder_layer in enumerate(self.layers):
      if output_hidden_states:
        encoder_states.append(x)
      # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
      dropout_probability = random.uniform(0, 1)
      if self.training and (dropout_probability < self.layerdrop):  # skip the layer
        attn = None
      else:
        x, attn = encoder_layer(x, attention_mask, output_attentions=output_attentions)

        hidden_states = x.permute(1, 0, 2)
        pooled_output = hidden_states[:, 0]  # [batch_size*max_utterance, dim]

        # perform GAT at last k layers
        if i >= self.gnn_layer_start and (not self.ignore_last_layer or i < self.num_layers-1):
          feats = {
              'utterance': pooled_output.clone(),  # [batch_size*num_nodes, dim]
              'speaker': speaker_hidden  # (bz*MAX_SPEAKER_NUM, hidden_dim)
          }
          _ids = i - self.gnn_layer_start
          graph_outputs = self.gat_layer[_ids](graph, feats)
          utterance_combine = self.combine_layer[_ids][self.node_dict['utterance']](torch.cat((graph_outputs['utterance'], feats['utterance']), -1))
          speaker_combine = self.combine_layer[_ids][self.node_dict['speaker']](torch.cat((graph_outputs['speaker'], feats['speaker']), -1))

          hidden_states[:, 0] = utterance_combine
          speaker_hidden = speaker_combine

      if output_attentions:
        all_attentions = all_attentions + (attn,)

    if self.layer_norm:
      x = self.layer_norm(x)
    if output_hidden_states:
      encoder_states.append(x)
      # T x B x C -> B x T x C
      encoder_states = tuple(hidden_state.transpose(0, 1) for hidden_state in encoder_states)

    # T x B x C -> B x T x C
    x = x.transpose(0, 1)

    if not return_dict:
      return tuple(v for v in [x, encoder_states, all_attentions] if v is not None)
    return BaseModelOutput(last_hidden_state=x, hidden_states=encoder_states, attentions=all_attentions)
