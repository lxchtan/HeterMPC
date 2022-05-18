"""
Campare with d2d_3:
  Decoder use six layer transformers.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from typing import List, Optional, Tuple

from transformers.utils import logging

from transformers.modeling_encoder_decoder import EncoderDecoderModel, EncoderDecoderConfig
from transformers.modeling_outputs import Seq2SeqLMOutput, ModelOutput
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_bert import BertPreTrainedModel, BertLMHeadModel

from dataloaders import MAX_UTTERANCE_NUM, MAX_SPEAKER_NUM
from models.HGT import bert_modify

logger = logging.get_logger(__name__)

class TXHGenerationModel(EncoderDecoderModel):
  def __init__(
      self,
      config: Optional[PretrainedConfig] = None,
      encoder: Optional[PreTrainedModel] = None,
      decoder: Optional[PreTrainedModel] = None,
      args=None,
  ):
    assert config is not None or (
        encoder is not None and decoder is not None
    ), "Either a configuration or an Encoder and a decoder has to be provided"
    if config is None:
      config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config)
    else:
      assert isinstance(config, self.config_class), "config: {} has to be of type {}".format(
          config, self.config_class
      )
    # initialize with config
    super(EncoderDecoderModel, self).__init__(config)

    # Newly define
    self.args = args
    self.config = config

    EncoderModel = bert_modify.TXHModel
    DecoderModel = BertLMHeadModel
    # End

    if encoder is None:
      encoder = EncoderModel(config.encoder, args, add_pooling_layer=False)

    if decoder is None:
      decoder = DecoderModel.from_config(config.decoder)

    self.encoder = encoder
    self.decoder = decoder
    assert (
        self.encoder.get_output_embeddings() is None
    ), "The encoder {} should not have a LM Head. Please use a model without LM Head"

    # tie encoder, decoder weights if config set accordingly
    self.tie_weights()

  def forward(
      self,
      input_ids=None,
      graph=None,
      attention_mask=None,
      decoder_input_ids=None,
      decoder_attention_mask=None,
      encoder_outputs=None,
      past_key_values=None,  # TODO: (PVP) implement :obj:`use_cache`
      inputs_embeds=None,
      decoder_inputs_embeds=None,
      labels=None,
      ans_idxs=None,
      use_cache=None,  # TODO: (PVP) implement :obj:`use_cache`
      output_attentions=None,
      output_hidden_states=None,
      return_dict=True,
      **kwargs,
  ):
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

    kwargs_decoder = {
        argument[len("decoder_"):]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
    }

    if encoder_outputs is None:
      encoder_outputs = self.encoder(
          graph,
          input_ids=input_ids,
          attention_mask=attention_mask,
          inputs_embeds=inputs_embeds,
          output_attentions=output_attentions,
          output_hidden_states=output_hidden_states,
          return_dict=return_dict,
          **kwargs_encoder,
      )

    sequence_output = encoder_outputs[0]

    cls_list = sequence_output.view(ans_idxs.shape[0], -1,  sequence_output.size(1), sequence_output.size(2))[:, :, 0, :].squeeze(2) #(bz, ns, dim)
    encoder_attention_mask_d = torch.arange(0, MAX_UTTERANCE_NUM, device=ans_idxs.device, dtype=ans_idxs.dtype).repeat(ans_idxs.shape[0], 1)
    encoder_attention_mask_d = (encoder_attention_mask_d < ans_idxs.unsqueeze(-1)).to(dtype=ans_idxs.dtype)

    # Decode
    # resp_node_id = ans_idxs + torch.arange(0, MAX_UTTERANCE_NUM * ans_idxs.shape[0], MAX_UTTERANCE_NUM, device=ans_idxs.device, dtype=ans_idxs.dtype)
    # response_hidden = sequence_output.index_select(dim=0, index=resp_node_id)
    response_hidden = cls_list.gather(dim=1, index=ans_idxs[:, None, None].expand(ans_idxs.size(0), 1, cls_list.size(-1))).squeeze(1)
    decoder_inputs_embeds = self.decoder.bert.embeddings.word_embeddings(decoder_input_ids)
    decoder_inputs_embeds[:, 0, :] = response_hidden

    decoder_outputs = self.decoder(
        # input_ids=decoder_input_ids,
        attention_mask=decoder_attention_mask,
        encoder_hidden_states=cls_list,
        encoder_attention_mask=encoder_attention_mask_d,
        inputs_embeds=decoder_inputs_embeds,
        labels=labels,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        **kwargs_decoder,
    )

    # TODO(PVP): currently it is not possible to use `past`
    if not return_dict:
      return decoder_outputs + encoder_outputs

    return Seq2SeqLMOutput(
        loss=decoder_outputs.loss,
        logits=decoder_outputs.logits,
        past_key_values=None,  # TODO(PVP) - need to implement cache for BERT, etc... before this works
        decoder_hidden_states=decoder_outputs.hidden_states,
        decoder_attentions=decoder_outputs.attentions,
        cross_attentions=decoder_outputs.cross_attentions,
        encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        encoder_hidden_states=encoder_outputs.hidden_states,
        encoder_attentions=encoder_outputs.attentions,
    )

  def _prepare_encoder_decoder_kwargs_for_generation(self, input_ids: torch.LongTensor, model_kwargs):
    # retrieve encoder hidden states
    encoder = self.get_encoder()
    encoder_kwargs = {
        argument: value for argument, value in model_kwargs.items() if not argument.startswith("decoder_")
    }
    graph = encoder_kwargs.pop('graph')
    ans_idxs = encoder_kwargs.pop('ans_idxs')

    encoder_outputs = encoder(graph, input_ids=input_ids, return_dict=True, **encoder_kwargs)

    model_kwargs["encoder_outputs"] = encoder_outputs

    return model_kwargs

  def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, encoder_outputs=None, ans_idxs=None,  **kwargs):
    decoder_inputs = self.decoder.prepare_inputs_for_generation(input_ids)
    decoder_attention_mask = decoder_inputs["attention_mask"] if "attention_mask" in decoder_inputs else None
    input_dict = {
        # "attention_mask": attention_mask,
        "decoder_attention_mask": decoder_attention_mask,
        "decoder_input_ids": decoder_inputs["input_ids"],
        "encoder_outputs": encoder_outputs,
        "ans_idxs": ans_idxs,
    }

    # Ideally all models should have a :obj:`use_cache`
    # leave following to ifs until all have it implemented
    if "use_cache" in decoder_inputs:
      input_dict["decoder_use_cache"] = decoder_inputs["use_cache"]

    if "past_key_values" in decoder_inputs:
      input_dict["past_key_values"] = decoder_inputs["past_key_values"]

    return input_dict

  def resize_token_embeddings(self, new_num_tokens: Optional[int] = None) -> torch.nn.Embedding:
    model_embeds = self.encoder.resize_token_embeddings(new_num_tokens)
    decoder_model_embeds = self.decoder.resize_token_embeddings(new_num_tokens)

    # Tie weights again if needed
    self.tie_weights()

    return model_embeds

  @classmethod
  def from_encoder_decoder_pretrained(
      cls,
      encoder_pretrained_model_name_or_path: str = None,
      decoder_pretrained_model_name_or_path: str = None,
      *model_args,
      **kwargs
  ) -> PreTrainedModel:
    kwargs_encoder = {
        argument[len("encoder_"):]: value for argument, value in kwargs.items() if argument.startswith("encoder_")
    }

    kwargs_decoder = {
        argument[len("decoder_"):]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
    }

    # remove encoder, decoder kwargs from kwargs
    for key in kwargs_encoder.keys():
      del kwargs["encoder_" + key]
    for key in kwargs_decoder.keys():
      del kwargs["decoder_" + key]

    # Load and initialize the encoder and decoder
    # The distinction between encoder and decoder at the model level is made
    # by the value of the flag `is_decoder` that we need to set correctly.
    EncoderModel = bert_modify.TXHModel
    DecoderModel = BertLMHeadModel

    encoder = kwargs_encoder.pop("model", None)
    if encoder is None:
      assert (
          encoder_pretrained_model_name_or_path is not None
      ), "If `model` is not defined as an argument, a `encoder_pretrained_model_name_or_path` has to be defined"

      if "config" not in kwargs_encoder:
        from transformers.configuration_auto import AutoConfig

        encoder_config = AutoConfig.from_pretrained(encoder_pretrained_model_name_or_path)
        if encoder_config.is_decoder is True or encoder_config.add_cross_attention is True:

          logger.info(
              f"Initializing {encoder_pretrained_model_name_or_path} as a encoder model from a decoder model. Cross-attention and casual mask are disabled."
          )
          encoder_config.is_decoder = False
          encoder_config.add_cross_attention = False

        kwargs_encoder["config"] = encoder_config

      encoder = EncoderModel.from_pretrained(encoder_pretrained_model_name_or_path, args=kwargs["args"], *model_args, **kwargs_encoder)

    decoder = kwargs_decoder.pop("model", None)
    if decoder is None:
      assert (
          decoder_pretrained_model_name_or_path is not None
      ), "If `decoder_model` is not defined as an argument, a `decoder_pretrained_model_name_or_path` has to be defined"

      if "config" not in kwargs_decoder:
        from transformers.configuration_auto import AutoConfig

        decoder_config = AutoConfig.from_pretrained(decoder_pretrained_model_name_or_path)
        if decoder_config.is_decoder is False or decoder_config.add_cross_attention is False:
          logger.info(
              f"Initializing {decoder_pretrained_model_name_or_path} as a decoder model. Cross attention layers are added to {decoder_pretrained_model_name_or_path} and randomly initialized if {decoder_pretrained_model_name_or_path}'s architecture allows for cross attention layers."
          )
          decoder_config.is_decoder = True
          decoder_config.add_cross_attention = True

        decoder_config.num_hidden_layers = 6
        kwargs_decoder["config"] = decoder_config

      if kwargs_decoder["config"].is_decoder is False or kwargs_decoder["config"].add_cross_attention is False:
        logger.warning(
            f"Decoder model {decoder_pretrained_model_name_or_path} is not initialized as a decoder. In order to initialize {decoder_pretrained_model_name_or_path} as a decoder, make sure that the attributes `is_decoder` and `add_cross_attention` of `decoder_config` passed to `.from_encoder_decoder_pretrained(...)` are set to `True` or do not pass a `decoder_config` to `.from_encoder_decoder_pretrained(...)`"
        )

      decoder = DecoderModel(**kwargs_decoder)

    # instantiate config with corresponding kwargs
    config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config, **kwargs)
    return cls(encoder=encoder, decoder=decoder, config=config, args=kwargs["args"])
