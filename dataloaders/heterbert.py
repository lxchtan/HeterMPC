"""
 Do not drop sentences like dataloader.
 Do not add encoding in graph.
"""
import torch
import os
import json
import logging

from itertools import chain
from copy import deepcopy
from utils.data import truncate_utterance, pad_ids, pad_ids_2D

from tqdm import tqdm
from collections import defaultdict

import dgl

from dataloaders import MAX_SPEAKER_NUM, MAX_UTTERANCE_NUM

logger = logging.getLogger(__name__)

SPECIAL_TOKENS_VALUES = ["[PAD]", "[CLS]", "[SEP]"]

class DatasetWalker(object):
  """
    {
      "context": [
        "should i try building an rpm or do i just write a bash script and run it sudo ?",
        "the common postfix setup routine handles it",
        "can i automate that . it asks me questions .",
        "set it up properly and do : tar -cvzf postfix.tgz FILEPATH",
        "but i would setup one machine with real credentials and all the other machines will use that relay as their relay",
        "i 'm trying to build an ec2 image or generate ubuntu FILEPATH for node.js developers ."
      ],
      "relation_at": [[1, 0], [2, 1], [3, 2], [4, 2], [5, 4]],
      "ctx_spk": [1, 2, 1, 2, 2, 1],
      "ctx_adr": [-1, 1, 2, 1, 1, 2],
      "answer": "and untar your default config on all your client machines .",
      "ans_idx": 2,
      "ans_spk": 2,
      "ans_adr": 1
    }
  """
  def __init__(self, dataset, dataroot, labels):
    self.labels = labels
    path = os.path.join(os.path.abspath(dataroot))

    if dataset not in ['train', 'valid', 'test']:
      raise ValueError('Wrong dataset name: %s' % (dataset))
    
    self.data = []
    file = os.path.join(path, f'ubuntu_data_{dataset}.json')

    with open(file, 'r') as f:
      for line in f:
        data_line = json.loads(line)
        history = data_line['context']
        relation = {
          'relation_at': data_line['relation_at'],
          'index': len(history),
          'from_idx': data_line['ans_idx'],
          'ctx_spk': [spk - 1 for spk in data_line['ctx_spk']],
          'ctx_adr': [spk - 1 for spk in data_line['ctx_adr']],
          'ans_spk': data_line['ans_spk'] - 1,
          'ans_adr': data_line['ans_adr'] - 1
        }
        label = {
          'response': data_line['answer'],
        }
        self.data.append((history, relation, label))

  def __iter__(self):
    for history, relation, label in self.data:
      yield history, relation, (label if self.labels else None)

  def __len__(self, ):
    return len(self.data)

class BaseDataset(torch.utils.data.Dataset):
  def __init__(self, args, tokenizer, split_type, labels=True):
    self.args = args
    self.dataroot = args.dataroot
    self.tokenizer = tokenizer
    self.split_type = split_type

    self.SPECIAL_TOKENS_VALUES = SPECIAL_TOKENS_VALUES
    self.bos = self.tokenizer.convert_tokens_to_ids("[CLS]")
    self.eos = self.tokenizer.convert_tokens_to_ids("[SEP]")
    self.pad = self.tokenizer.convert_tokens_to_ids("[PAD]")

    if not os.path.exists(args.dataset_cache):
      os.makedirs(args.dataset_cache)
    
    data_cache_uniflag = "" if not getattr(self.args, 'unidirected', False) else "_unidirected"
    data_cache = os.path.join(args.dataset_cache, f'{split_type}_{args.dataloader}_{args.dataloader_class}{data_cache_uniflag}.pt')
    if os.path.exists(data_cache):
      logger.info("Loading examples from cache.")
      self.examples = torch.load(data_cache)
    else:
      self.all_response_tokenized = []
      self.dataset_walker = DatasetWalker(split_type, labels=labels, dataroot=self.dataroot)
      self.dialogs = self._prepare_conversations()
      self._create_examples()

      torch.save(self.examples, data_cache)

  def _prepare_conversations(self):
    logger.info("Construting Graph")
    tokenized_dialogs = []
    for history, relation, label in tqdm(self.dataset_walker, disable=self.args.local_rank not in [-1, 0], desc="Construting Graph"):  # only show progress bar in one process
      dialog = {}
      dialog["log"] = history
      dialog["answer_index"] = relation['index']
      
      # history utterance
      edges = tuple(map(lambda x:x.squeeze(1), torch.tensor(relation['relation_at']).split(1, dim=-1)))
      inv_edges = deepcopy((edges[1], edges[0]))
      
      # history speaker
      speaker_to_utterance = []
      speaker_to_utterance_inv = []
      utterance_addr_speaker = []
      utterance_addr_speaker_inv = []
      for idx, spk in enumerate(relation['ctx_spk']):
        speaker_to_utterance.append([spk, idx])
        speaker_to_utterance_inv.append([idx, spk])
      for idx, adr in enumerate(relation['ctx_adr']):
        if adr == -2: continue
        utterance_addr_speaker_inv.append([adr, idx])
        utterance_addr_speaker.append([idx, adr])

      data_dict = {
        ('utterance', 'from', 'utterance'): edges,
        ('utterance', 'to', 'utterance'): inv_edges,
        ('speaker', 'speak', 'utterance'): speaker_to_utterance,
        ('utterance', 'speaked', 'speaker'): speaker_to_utterance_inv,
        ('utterance', 'aim_to', 'speaker'): utterance_addr_speaker,
        ('speaker', 'get_from', 'utterance'): utterance_addr_speaker_inv,
      }
      graph = dgl.heterograph(data_dict, num_nodes_dict={
        'utterance': MAX_UTTERANCE_NUM,
        'speaker': MAX_SPEAKER_NUM
      })
      # response 
      graph.add_edges(relation['from_idx'], relation['index'], etype='to')
      graph.add_edges(relation['ans_spk'], relation['index'], etype='speak')
      graph.add_edges(relation['ans_adr'], relation['index'],  etype='get_from')

      dialog['graph'] = graph
      dialog["label"] = label

      tokenized_dialogs.append(dialog)
    return tokenized_dialogs

  def _create_examples(self):
    logger.info("Creating examples")
    self.examples = []
    for dialog in tqdm(self.dialogs, disable=self.args.local_rank not in [-1, 0], desc="Creating examples"):
      label = dialog["label"]
      graph = dialog["graph"]
      ans_index = dialog['answer_index']
      dialog = dialog["log"]

      history = [
        self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(turn))
        for turn in dialog
      ]
      gt_resp = label.get("response", "")
      tokenized_gt_resp = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(gt_resp))

      # apply history threshold at an utterance-level (a large value can be used to nullify its effect)
      truncated_history = history[-self.args.history_max_utterances:]

      # perform token-level truncation of history from the left
      truncated_history = truncate_utterance(truncated_history, self.args.utterance_max_tokens - 2)

      self.examples.append({
        "graph": graph,
        "history": truncated_history,
        "response": tokenized_gt_resp,
        "response_text": gt_resp,
        'answer_index': ans_index,
        "label": label,
      })

  def build_graph_from_examples(self, graph, history, response, with_eos=True):
    input_datas = {}
    input_encoding = []
    input_segment_id = []
    input_encoding_mask = []
    # Cut Response
    response = response[:self.args.utterance_max_tokens-2]

    for i, h in enumerate(history):
      sequence = [self.bos] + h + [self.eos]
      sequence_len = len(sequence)
      input_encoding.append(torch.tensor(sequence + [self.pad] * (self.args.utterance_max_tokens - sequence_len)).unsqueeze(0)) # [1, utterance_max_tokens]
      input_segment_id.append(torch.zeros((1, self.args.utterance_max_tokens,), dtype=torch.long)) # [1, utterance_max_tokens]
      my_mask = torch.zeros((self.args.utterance_max_tokens, self.args.utterance_max_tokens), dtype=torch.long)
      my_mask[:sequence_len, :sequence_len] = 1
      input_encoding_mask.append(my_mask.unsqueeze(0)) # [1, utterance_max_tokens, utterance_max_tokens]
    
    # Response
    response_id = len(history)
    sequence = [self.bos] + response + ([self.eos] if with_eos else [])
    sequence_len = len(sequence)
    decode_input = sequence
    decoder_attention_mask = torch.tril(torch.ones((sequence_len, sequence_len), dtype=torch.long)).numpy().tolist()

    input_encoding.append(torch.tensor([self.bos] + [self.pad] * (self.args.utterance_max_tokens - 1)).unsqueeze(0)) # [1, utterance_max_tokens]
    input_segment_id.append(torch.zeros((1, self.args.utterance_max_tokens), dtype=torch.long)) # [1, utterance_max_tokens]
    my_mask = torch.zeros((self.args.utterance_max_tokens, self.args.utterance_max_tokens), dtype=torch.long)
    my_mask[0, 0] = 1
    input_encoding_mask.append(my_mask.unsqueeze(0)) # [1, utterance_max_tokens, utterance_max_tokens]

    # input_encoding.append(torch.tensor(sequence + [self.pad] * (self.args.utterance_max_tokens - sequence_len)).unsqueeze(0)) # [1, utterance_max_tokens]
    # input_segment_id.append(torch.zeros((1, self.args.utterance_max_tokens), dtype=torch.long)) # [1, utterance_max_tokens]
    # my_mask = torch.zeros((self.args.utterance_max_tokens, self.args.utterance_max_tokens), dtype=torch.long)
    # my_mask[:sequence_len, :sequence_len] = torch.tril(torch.ones((sequence_len, sequence_len), dtype=torch.long))
    # input_encoding_mask.append(my_mask.unsqueeze(0)) # [1, utterance_max_tokens, utterance_max_tokens]

    response_gt = [-100] + response + [self.eos]

    input_datas['encoding'] = torch.cat(input_encoding, dim=0)
    input_datas['segment_id'] = torch.cat(input_segment_id, dim=0)
    input_datas['encoding_mask'] = torch.cat(input_encoding_mask, dim=0)

    addition_nums = MAX_UTTERANCE_NUM - response_id - 1
    if addition_nums != 0:
      input_datas['encoding'] = torch.cat([input_datas['encoding'], torch.zeros((addition_nums, self.args.utterance_max_tokens), dtype=torch.long).fill_(self.pad)], dim=0)
      input_datas['segment_id'] = torch.cat([input_datas['segment_id'], torch.zeros((addition_nums, self.args.utterance_max_tokens), dtype=torch.long)], dim=0)
      input_datas['encoding_mask'] = torch.cat([input_datas['encoding_mask'], torch.zeros((addition_nums, self.args.utterance_max_tokens, self.args.utterance_max_tokens), dtype=torch.long)], dim=0)

    instance = {
      'graph': graph,
      'input_datas': input_datas,
      'decode_input': decode_input,  
      'decoder_attention_mask': decoder_attention_mask,
      'response': response_gt,
      'answer_index': response_id
    }

    return instance

  def __getitem__(self, index):
    example = self.examples[index]
    instance = self.build_graph_from_examples(
      example["graph"],
      example["history"],
      example["response"]
    )
    return instance

  def __len__(self):
    return len(self.examples)

  def collate_fn(self, batch):
    graph = [ins['graph'] for ins in batch]
    encoding = [ins['input_datas']['encoding'] for ins in batch]
    segment_id = [ins['input_datas']['segment_id'] for ins in batch]
    encoding_mask = [ins['input_datas']['encoding_mask'] for ins in batch] 
    decode_input = [ins['decode_input'] for ins in batch]   
    decoder_attention_mask = [ins['decoder_attention_mask'] for ins in batch]
    response = [ins['response'] for ins in batch]
    answer_index = [ins['answer_index'] for ins in batch]

    graph = dgl.batch(graph)
    encoding = torch.cat(encoding, dim=0)
    segment_id = torch.cat(segment_id, dim=0)
    encoding_mask = torch.cat(encoding_mask, dim=0)
    decode_input = torch.tensor(pad_ids(decode_input, self.pad))
    decoder_attention_mask = torch.tensor(pad_ids_2D(decoder_attention_mask, 0))
    response = torch.tensor(pad_ids(response, self.pad))
    answer_index = torch.tensor(answer_index)

    return graph, encoding, decode_input, segment_id, encoding_mask, decoder_attention_mask, response, answer_index

class testDataset(BaseDataset):
  def __getitem__(self, index):
    example = self.examples[index]
    return example

  def collate_fn(self, batch):
    return batch
  
  def collate_step(self, args, graph, history, current_output):
    instance = self.build_graph_from_examples(
      graph, history, current_output, with_eos=False
    )
    new_graph = instance['graph'].to(args.device)
    ans_idxs = torch.tensor([instance['answer_index']]).to(args.device)

    input_ids = instance['input_datas']['encoding'].to(args.device)
    input_masks = instance['input_datas']['encoding_mask'].to(args.device)
    segment_ids = instance['input_datas']['segment_id'].to(args.device)

    return new_graph, input_ids, segment_ids, input_masks, ans_idxs