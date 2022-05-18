import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_bert import (
    BertEncoder,
    BertLayer,
    BertModel,
    BaseModelOutputWithCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions
)

from .HGT import HGTLayer
from dataloaders import MAX_UTTERANCE_NUM, MAX_SPEAKER_NUM


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


class TXHEncoder(BertEncoder):
  def __init__(self, config, args):
    super().__init__(config)
    self.config = config
    node_dict = {"utterance":0, "speaker": 1}
    edge_dict = {'from':0, 'to':1, 'speak': 2, 'speaked': 3, 'aim_to':4, 'get_from':5}
    self.node_dict = node_dict
    self.edge_dict = edge_dict
    
    self.gnn_layer_start = args.gnn_layer_start
    self.ignore_last_layer = getattr(args, "ignore_last_layer", False)
    self.num_layers = config.num_hidden_layers
    self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
    self.gat_layer = nn.ModuleList([HGTModel(config, args, node_dict, edge_dict) for _ in range(self.num_layers - self.gnn_layer_start)])
    self.combine_layer = nn.ModuleList([nn.ModuleList([
      nn.Linear(config.hidden_size * 2, config.hidden_size)
      for _ in range(len(node_dict))
    ]) for _ in range(self.num_layers - self.gnn_layer_start)])
    
    self.speakers = nn.Embedding(MAX_SPEAKER_NUM, config.hidden_size)

  def forward(
      self,
      graph,
      hidden_states,
      attention_mask=None,
      head_mask=None,
      encoder_hidden_states=None,
      encoder_attention_mask=None,
      output_attentions=False,
      output_hidden_states=False,
      return_dict=False,
  ):
    all_hidden_states = () if output_hidden_states else None
    all_self_attentions = () if output_attentions else None
    all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

    batch_size = hidden_states.size(0)//MAX_UTTERANCE_NUM
    speaker_hidden = self.speakers(torch.arange(0, MAX_SPEAKER_NUM, dtype=torch.long, device=hidden_states.device
                      ).view(1,-1).repeat(batch_size, 1)).view(-1, self.config.hidden_size) # (bz*MAX_SPEAKER_NUM, hidden_dim)

    for i, layer_module in enumerate(self.layer):
      if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

      layer_head_mask = head_mask[i] if head_mask is not None else None
      layer_outputs = layer_module(
          hidden_states,
          attention_mask,
          layer_head_mask,
          encoder_hidden_states,
          encoder_attention_mask,
          output_attentions,
      )
      hidden_states = layer_outputs[0]
      pooled_output = hidden_states[:, 0] # [batch_size*max_utterance, dim]

      # perform GAT at last k layers 
      if i >= self.gnn_layer_start and (not self.ignore_last_layer or i < self.num_layers-1):
        feats = {
          'utterance': pooled_output.clone(), # [batch_size*num_nodes, dim]
          'speaker': speaker_hidden # (bz*MAX_SPEAKER_NUM, hidden_dim)
        } 
        _ids = i - self.gnn_layer_start
        graph_outputs = self.gat_layer[_ids](graph, feats)
        utterance_combine = self.combine_layer[_ids][self.node_dict['utterance']](torch.cat((graph_outputs['utterance'], feats['utterance']), -1))
        speaker_combine = self.combine_layer[_ids][self.node_dict['speaker']](torch.cat((graph_outputs['speaker'], feats['speaker']), -1))

        hidden_states[:, 0] = utterance_combine
        speaker_hidden = speaker_combine

      if output_attentions:
        all_self_attentions = all_self_attentions + (layer_outputs[1],)
        if self.config.add_cross_attention:
          all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

    if output_hidden_states:
      all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
      return tuple(
          v
          for v in [hidden_states, all_hidden_states, all_self_attentions, all_cross_attentions]
          if v is not None
      )
    return BaseModelOutputWithCrossAttentions(
        last_hidden_state=hidden_states,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
        cross_attentions=all_cross_attentions,
    )


class TXHModel(BertModel):
  def __init__(self, config, args, add_pooling_layer=True):
    super().__init__(config, add_pooling_layer)
    self.encoder = TXHEncoder(config, args)
    self.init_weights()

  def forward(
      self,
      graph,
      input_ids=None,
      attention_mask=None,
      token_type_ids=None,
      position_ids=None,
      head_mask=None,
      inputs_embeds=None,
      encoder_hidden_states=None,
      encoder_attention_mask=None,
      output_attentions=None,
      output_hidden_states=None,
      return_dict=None,
  ):
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
      raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
      input_shape = input_ids.size()
    elif inputs_embeds is not None:
      input_shape = inputs_embeds.size()[:-1]
    else:
      raise ValueError("You have to specify either input_ids or inputs_embeds")

    device = input_ids.device if input_ids is not None else inputs_embeds.device

    if attention_mask is None:
      attention_mask = torch.ones(input_shape, device=device)
    if token_type_ids is None:
      token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

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

    embedding_output = self.embeddings(
        input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
    )
    encoder_outputs = self.encoder(
        graph,
        embedding_output,
        attention_mask=extended_attention_mask,
        head_mask=head_mask,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_extended_attention_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    sequence_output = encoder_outputs[0]
    pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

    if not return_dict:
      return (sequence_output, pooled_output) + encoder_outputs[1:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        last_hidden_state=sequence_output,
        pooler_output=pooled_output,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
        cross_attentions=encoder_outputs.cross_attentions,
    )
