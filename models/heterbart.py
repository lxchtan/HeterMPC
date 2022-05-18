import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

import warnings

from transformers.modeling_bart import (
    BartModel,
    BartDecoder,
    BartForConditionalGeneration,
    _prepare_bart_decoder_inputs,
    shift_tokens_right,
    BaseModelOutput,
    Seq2SeqModelOutput,
    Seq2SeqLMOutput
)

from dataloaders import MAX_UTTERANCE_NUM, MAX_SPEAKER_NUM
from models.HGT import bart_modify

class TXHModel(BartModel):
  def __init__(self, config, args):
    super(BartModel, self).__init__(config)

    padding_idx, vocab_size = config.pad_token_id, config.vocab_size
    self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

    self.encoder = bart_modify.TXHEncoder(config, self.shared, args)
    self.decoder = BartDecoder(config, self.shared)

    self.ans_embeddings = nn.Embedding(3,  config.d_model)

    self.init_weights()

  def forward(
      self,
      input_ids,
      graph,
      attention_mask=None,
      decoder_input_ids=None,
      decoder_attention_mask=None,
      encoder_outputs=None,
      decoder_ans_idxs=None,
      decoder_ans_from=None,
      past_key_values=None,
      use_cache=None,
      output_attentions=None,
      output_hidden_states=None,
      return_dict=None,
      **kwargs,
  ):
    if "decoder_past_key_values" in kwargs:
      warnings.warn(
          "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
          FutureWarning,
      )
      past_key_values = kwargs.pop("decoder_past_key_values")

    if decoder_input_ids is None:
      use_cache = False

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # make masks if user doesn't supply
    if not use_cache:
      decoder_input_ids, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
          self.config,
          input_ids,
          decoder_input_ids=decoder_input_ids,
          decoder_padding_mask=decoder_attention_mask,
          causal_mask_dtype=self.shared.weight.dtype,
      )
    else:
      decoder_padding_mask, causal_mask = None, None

    assert decoder_input_ids is not None

    if encoder_outputs is None:
      encoder_outputs = self.encoder(
          graph=graph,
          input_ids=input_ids,
          attention_mask=attention_mask,
          output_attentions=output_attentions,
          output_hidden_states=output_hidden_states,
          return_dict=return_dict,
      )
    # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=False
    elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
      encoder_outputs = BaseModelOutput(
          last_hidden_state=encoder_outputs[0],
          hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
          attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
      )

    # deal with NAN for invaild example. 
    # encoder_outputs[0][encoder_outputs[0].isnan()] = 0.0
    encoder_output_embeddings = encoder_outputs[0].reshape(-1, MAX_UTTERANCE_NUM*encoder_outputs[0].shape[1], encoder_outputs[0].shape[2])
    
    ans_total = torch.zeros(encoder_outputs[0].shape[0], dtype=decoder_ans_idxs.dtype, device=decoder_ans_idxs.device).view(-1, MAX_UTTERANCE_NUM)
    ans_total = ans_total.scatter_add(dim=1, index=decoder_ans_idxs.unsqueeze(1), src=torch.ones_like(ans_total)+1).scatter_add(dim=1, index=decoder_ans_from.unsqueeze(1), src=torch.ones_like(ans_total))
    ans_embeddings = self.ans_embeddings(ans_total).unsqueeze(2).repeat(1, 1, encoder_outputs[0].shape[1], 1).view_as(encoder_output_embeddings)

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    decoder_outputs = self.decoder(
        decoder_input_ids,
        encoder_output_embeddings + ans_embeddings,
        attention_mask.reshape(-1, MAX_UTTERANCE_NUM*attention_mask.shape[1]) if attention_mask is not None else None,
        decoder_padding_mask,
        decoder_causal_mask=causal_mask,
        past_key_values=past_key_values,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    if not return_dict:
      return decoder_outputs + encoder_outputs

    return Seq2SeqModelOutput(
        last_hidden_state=decoder_outputs.last_hidden_state,
        past_key_values=decoder_outputs.past_key_values,
        decoder_hidden_states=decoder_outputs.hidden_states,
        decoder_attentions=decoder_outputs.attentions,
        cross_attentions=decoder_outputs.cross_attentions,
        encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        encoder_hidden_states=encoder_outputs.hidden_states,
        encoder_attentions=encoder_outputs.attentions,
    )


class TXHGenerationModel(BartForConditionalGeneration):

  def __init__(self, config, args):
    super(BartForConditionalGeneration, self).__init__(config)
    base_model = TXHModel(config, args)
    self.model = base_model
    self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))

  def forward(
      self,
      input_ids,
      graph=None,
      attention_mask=None,
      decoder_input_ids=None,
      decoder_attention_mask=None,
      encoder_outputs=None,
      past_key_values=None,
      labels=None,
      decoder_ans_idxs=None,
      decoder_ans_from=None,
      use_cache=None,
      output_attentions=None,
      output_hidden_states=None,
      return_dict=None,
      **unused,
  ):
    r"""
    labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
        Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
        config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
        (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

    Returns:

    Conditional generation example::

        >>> # Mask filling only works for bart-large
        >>> from transformers import BartTokenizer, BartForConditionalGeneration
        >>> tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        >>> TXT = "My friends are <mask> but they eat too many carbs."

        >>> model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        >>> input_ids = tokenizer([TXT], return_tensors='pt')['input_ids']
        >>> logits = model(input_ids).logits

        >>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
        >>> probs = logits[0, masked_index].softmax(dim=0)
        >>> values, predictions = probs.topk(5)

        >>> tokenizer.decode(predictions).split()
        >>> # ['good', 'great', 'all', 'really', 'very']
    """
    if "lm_labels" in unused:
      warnings.warn(
          "The `lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
          FutureWarning,
      )
      labels = unused.pop("lm_labels")
    if "decoder_cached_states" in unused:
      warnings.warn(
          "The `decoder_cached_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
          FutureWarning,
      )
      past_key_values = unused.pop("decoder_cached_states")
    if "decoder_past_key_values" in unused:
      warnings.warn(
          "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
          FutureWarning,
      )
      past_key_values = unused.pop("decoder_past_key_values")
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if labels is not None:
      use_cache = False
      if decoder_input_ids is None:
        decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id)

    outputs = self.model(
        input_ids,
        graph,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        encoder_outputs=encoder_outputs,
        decoder_attention_mask=decoder_attention_mask,
        decoder_ans_idxs=decoder_ans_idxs,
        decoder_ans_from=decoder_ans_from,
        past_key_values=past_key_values,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)

    masked_lm_loss = None
    if labels is not None:
      loss_fct = CrossEntropyLoss()
      # TODO(SS): do we need to ignore pad tokens in labels?
      masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

    if not return_dict:
      output = (lm_logits,) + outputs[1:]
      return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

    return Seq2SeqLMOutput(
        loss=masked_lm_loss,
        logits=lm_logits,
        past_key_values=outputs.past_key_values,
        decoder_hidden_states=outputs.decoder_hidden_states,
        decoder_attentions=outputs.decoder_attentions,
        cross_attentions=outputs.cross_attentions,
        encoder_last_hidden_state=outputs.encoder_last_hidden_state,
        encoder_hidden_states=outputs.encoder_hidden_states,
        encoder_attentions=outputs.encoder_attentions,
    )
      
      
  def prepare_inputs_for_generation(
      self, decoder_input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, graph=None, decoder_ans_idxs=None, decoder_ans_from=None, **kwargs
  ):
    return {
        "input_ids": None,  # encoder_outputs is defined. input_ids not needed
        "graph": graph,
        "encoder_outputs": encoder_outputs,
        "past_key_values": past,
        "decoder_input_ids": decoder_input_ids,
        "attention_mask": attention_mask,
        "decoder_ans_idxs": decoder_ans_idxs,
        "decoder_ans_from": decoder_ans_from,
        "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
    }

  def _prepare_decoder_input_ids_for_generation(
        self, input_ids: torch.LongTensor, decoder_start_token_id: int = None, bos_token_id: int = None, **model_kwargs
  ) -> torch.LongTensor:

    if "decoder_input_ids" in model_kwargs:
        return model_kwargs["decoder_input_ids"]

    decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
    decoder_input_ids = (
        torch.ones((input_ids.shape[0]//MAX_UTTERANCE_NUM, 1), dtype=input_ids.dtype, device=input_ids.device)
        * decoder_start_token_id
    )
    return decoder_input_ids