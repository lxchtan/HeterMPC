"""
 Do not drop sentences like dataloader.
 Do not add encoding in graph.
"""
import torch
import os
import json
import logging

from utils.data import truncate_utterance, pad_ids

from tqdm import tqdm

import dgl

logger = logging.getLogger(__name__)

SPECIAL_TOKENS_VALUES = ["<pad>", "<s>", "</s>"]


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

    def __init__(self, labels, dataset, dataroot, batch_list):
        self.labels = labels

        path = os.path.join(os.path.abspath(dataroot))

        if dataset not in ['train', 'valid', 'test']:
            raise ValueError('Wrong dataset name: %s' % (dataset))

        if dataset in ['valid', 'test']:
            self.data = []
            file = os.path.join(path, f'ubuntu_data_{dataset}.json')
            # file = os.path.join(path, f'5_{dataset}.json')
            # file = os.path.join(path, f'10_{dataset}.json')
            # file = os.path.join(path, f'15_{dataset}.json')

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
                    label = {'response': data_line['answer'],}
                    self.data.append((history, relation, label))

        elif dataset in ['train']:   ###对列表数据进行处理,EM通过list的形式保存数据——>迭代中每步更改数据
            self.data = []
            if len(batch_list) > 0:
                for data_line in batch_list:
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
                    label = {'response': data_line['answer'],}
                    self.data.append((history, relation, label))
            else:
                raise ValueError('Wrong len batch_list: %s' % (len(batch_list)))

        else:
            raise ValueError('Wrong dataset name: %s' % (dataset))


    def __iter__(self):
        for history, relation, label in self.data:
            yield history, relation, (label if self.labels else None)

    def __len__(self, ):
        return len(self.data)


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args, tokenizer, split_type, line_batch_list, labels=True):
        self.args = args
        self.dataroot = args.dataroot
        self.tokenizer = tokenizer
        self.split_type = split_type
        self.line_batch_list = line_batch_list

        self.MAX_SPEAKER_NUM = args.MAX_SPEAKER_NUM
        self.MAX_UTTERANCE_NUM = args.MAX_UTTERANCE_NUM
        self.SPECIAL_TOKENS_VALUES = SPECIAL_TOKENS_VALUES
        self.bos = self.tokenizer.convert_tokens_to_ids("<s>")
        self.eos = self.tokenizer.convert_tokens_to_ids("</s>")
        self.pad = self.tokenizer.convert_tokens_to_ids("<pad>")

        if not os.path.exists(args.dataset_cache):
            os.makedirs(args.dataset_cache)

        data_cache_uniflag = "" if not getattr(self.args, 'unidirected', False) else "_unidirected"
        data_cache = os.path.join(args.dataset_cache,
                                  f'{split_type}_{args.dataloader}_{args.dataloader_class}{data_cache_uniflag}.pt')

        if split_type in ['test', 'valid']:  ####之所以这样是因为在EM迭代过程中test,dev是固定的,不需要更新数据，所以保存数据
            if os.path.exists(data_cache):
                logger.info("Loading examples from cache.")
                self.examples = torch.load(data_cache)
            else:
                self.all_response_tokenized = []
                self.dataset_walker = DatasetWalker(labels=labels, dataset=self.split_type, dataroot=self.dataroot,batch_list=self.line_batch_list)
                self.dialogs = self._prepare_conversations()
                self._create_examples()

                torch.save(self.examples, data_cache)

        else:   ####而训练数据随着迭代次数的变换在不停的改变，故不保存
            self.all_response_tokenized = []
            self.em_dataset_walker = DatasetWalker(labels=labels, dataset=self.split_type, dataroot=self.dataroot, batch_list=self.line_batch_list)
            self.em_dialogs = self._prepare_conversations_em()
            self._create_examples_em()



    def _prepare_conversations(self):
        logger.info("Construting Graph")
        tokenized_dialogs = []
        for history, relation, label in tqdm(self.dataset_walker, disable=self.args.local_rank not in [-1, 0],
                                             desc="Construting Graph"):  # only show progress bar in one process
            dialog = {}
            dialog["log"] = history
            dialog["answer_index"] = relation['index']
            dialog["answer_from"] = relation['from_idx']

            # history utterance
            # edges = tuple(map(lambda x:x.squeeze(1), torch.tensor(relation['relation_at']).split(1, dim=-1)))
            # inv_edges = deepcopy((edges[1], edges[0]))
            edges = relation['relation_at']
            inv_edges = [[e[1], e[0]] for e in edges]

            non_edges = []
            non_inv_edges = []

            total_u = len(history)
            for i in range(total_u):
                for j in range(i + 1, total_u):
                    if [j, i] not in relation['relation_at']:
                        non_edges.append([j, i])
                        non_inv_edges.append([i, j])

            # history speaker
            speaker_to_utterance = []
            speaker_to_utterance_inv = []
            utterance_addr_speaker = []
            utterance_addr_speaker_inv = []
            non_addr = []
            non_addr_inv = []
            for idx, spk in enumerate(relation['ctx_spk']):
                speaker_to_utterance.append([spk, idx])
                speaker_to_utterance_inv.append([idx, spk])
            for idx, adr in enumerate(relation['ctx_adr']):
                if adr == -2: continue
                utterance_addr_speaker_inv.append([adr, idx])
                utterance_addr_speaker.append([idx, adr])

            for u in range(total_u):
                for s in set(relation['ctx_spk']):
                    if [u, s] not in utterance_addr_speaker + speaker_to_utterance_inv:
                        non_addr.append([u, s])
                        non_addr_inv.append([s, u])

            # Response
            for i in range(total_u):
                if i != relation['from_idx']:
                    non_edges.append([relation['index'], i])
                    non_inv_edges.append([i, relation['index']])

            for s in set(relation['ctx_spk']):
                if s != relation['ans_spk'] and s != relation['ans_adr']:
                    non_addr.append([relation['index'], s])
                    non_addr_inv.append([s, relation['index']])

            data_dict = {
                ('utterance', 'from', 'utterance'): edges,
                ('utterance', 'to', 'utterance'): inv_edges,
                ('speaker', 'speak', 'utterance'): speaker_to_utterance,
                ('utterance', 'speaked', 'speaker'): speaker_to_utterance_inv,
                ('utterance', 'aim_to', 'speaker'): utterance_addr_speaker,
                ('speaker', 'get_from', 'utterance'): utterance_addr_speaker_inv,

                ('utterance', 'non_from', 'utterance'): non_edges,
                ('utterance', 'non_to', 'utterance'): non_inv_edges,
                ('utterance', 'non_aim_to', 'speaker'): non_addr,
                ('speaker', 'non_get_from', 'utterance'): non_addr_inv,
            }
            graph = dgl.heterograph(data_dict, num_nodes_dict={
                'utterance': self.MAX_UTTERANCE_NUM,
                'speaker': self.MAX_SPEAKER_NUM
            })
            # response
            graph.add_edges(relation['from_idx'], relation['index'], etype='to')
            graph.add_edges(relation['ans_spk'], relation['index'], etype='speak')
            graph.add_edges(relation['ans_adr'], relation['index'], etype='get_from')

            graph.add_edges(relation['index'], relation['from_idx'], etype='from')
            graph.add_edges(relation['index'], relation['ans_spk'], etype='speaked')
            graph.add_edges(relation['index'], relation['ans_adr'], etype='aim_to')

            dialog['graph'] = graph
            dialog["label"] = label

            tokenized_dialogs.append(dialog)
        return tokenized_dialogs

    def _prepare_conversations_em(self):
        # logger.info("Construting Graph")
        tokenized_dialogs = []
        for history, relation, label in self.em_dataset_walker:  # only show progress bar in one process
            dialog = {}
            dialog["log"] = history
            dialog["answer_index"] = relation['index']
            dialog["answer_from"] = relation['from_idx']

            # history utterance
            # edges = tuple(map(lambda x:x.squeeze(1), torch.tensor(relation['relation_at']).split(1, dim=-1)))
            # inv_edges = deepcopy((edges[1], edges[0]))
            edges = relation['relation_at']
            inv_edges = [[e[1], e[0]] for e in edges]

            non_edges = []
            non_inv_edges = []

            total_u = len(history)
            for i in range(total_u):
                for j in range(i + 1, total_u):
                    if [j, i] not in relation['relation_at']:
                        non_edges.append([j, i])
                        non_inv_edges.append([i, j])

            # history speaker
            speaker_to_utterance = []
            speaker_to_utterance_inv = []
            utterance_addr_speaker = []
            utterance_addr_speaker_inv = []
            non_addr = []
            non_addr_inv = []
            for idx, spk in enumerate(relation['ctx_spk']):
                speaker_to_utterance.append([spk, idx])
                speaker_to_utterance_inv.append([idx, spk])
            for idx, adr in enumerate(relation['ctx_adr']):
                if adr == -2: continue
                utterance_addr_speaker_inv.append([adr, idx])
                utterance_addr_speaker.append([idx, adr])

            for u in range(total_u):
                for s in set(relation['ctx_spk']):
                    if [u, s] not in utterance_addr_speaker + speaker_to_utterance_inv:
                        non_addr.append([u, s])
                        non_addr_inv.append([s, u])

            # Response
            for i in range(total_u):
                if i != relation['from_idx']:
                    non_edges.append([relation['index'], i])
                    non_inv_edges.append([i, relation['index']])

            for s in set(relation['ctx_spk']):
                if s != relation['ans_spk'] and s != relation['ans_adr']:
                    non_addr.append([relation['index'], s])
                    non_addr_inv.append([s, relation['index']])

            data_dict = {
                ('utterance', 'from', 'utterance'): edges,
                ('utterance', 'to', 'utterance'): inv_edges,
                ('speaker', 'speak', 'utterance'): speaker_to_utterance,
                ('utterance', 'speaked', 'speaker'): speaker_to_utterance_inv,
                ('utterance', 'aim_to', 'speaker'): utterance_addr_speaker,
                ('speaker', 'get_from', 'utterance'): utterance_addr_speaker_inv,

                ('utterance', 'non_from', 'utterance'): non_edges,
                ('utterance', 'non_to', 'utterance'): non_inv_edges,
                ('utterance', 'non_aim_to', 'speaker'): non_addr,
                ('speaker', 'non_get_from', 'utterance'): non_addr_inv,
            }
            graph = dgl.heterograph(data_dict, num_nodes_dict={
                'utterance': self.MAX_UTTERANCE_NUM,
                'speaker': self.MAX_SPEAKER_NUM
            })
            # response
            graph.add_edges(relation['from_idx'], relation['index'], etype='to')
            graph.add_edges(relation['ans_spk'], relation['index'], etype='speak')
            graph.add_edges(relation['ans_adr'], relation['index'], etype='get_from')

            graph.add_edges(relation['index'], relation['from_idx'], etype='from')
            graph.add_edges(relation['index'], relation['ans_spk'], etype='speaked')
            graph.add_edges(relation['index'], relation['ans_adr'], etype='aim_to')

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
            ans_from = dialog['answer_from']
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
                'answer_from': ans_from,
                "label": label,
            })

    def _create_examples_em(self):
        # logger.info("Creating examples")
        self.examples = []
        for dialog in self.em_dialogs:
            label = dialog["label"]
            graph = dialog["graph"]
            ans_index = dialog['answer_index']
            ans_from = dialog['answer_from']
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
                'answer_from': ans_from,
                "label": label,
            })

    def build_graph_from_examples(self, example, with_eos=True):
        graph = example["graph"]
        history = example["history"]
        response = example["response"]

        input_datas = {}
        input_encoding = []
        input_segment_id = []
        input_encoding_mask = []
        # Cut Response
        response = response[:self.args.utterance_max_tokens - 2]
        # response = response[:self.args.utterance_max_tokens - 1]

        for i, h in enumerate(history):
            sequence = [self.bos] + h + [self.eos]
            sequence_len = len(sequence)
            input_encoding.append(
                torch.tensor(sequence + [self.pad] * (self.args.utterance_max_tokens - sequence_len)).unsqueeze(
                    0))  # [1, utterance_max_tokens]
            input_segment_id.append(
                torch.zeros((1, self.args.utterance_max_tokens,), dtype=torch.long))  # [1, utterance_max_tokens]
            my_mask = torch.zeros(self.args.utterance_max_tokens, dtype=torch.long)
            my_mask[:sequence_len] = 1
            input_encoding_mask.append(my_mask.unsqueeze(0))  # [1, utterance_max_tokens]

        # Response
        response_id = len(history)
        sequence = [self.bos] + response + ([self.eos] if with_eos else [])
        sequence_len = len(sequence)
        decode_input = sequence

        input_encoding.append(torch.tensor([self.bos] + [self.pad] * (self.args.utterance_max_tokens - 1)).unsqueeze(
            0))  # [1, utterance_max_tokens]
        input_segment_id.append(
            torch.zeros((1, self.args.utterance_max_tokens), dtype=torch.long))  # [1, utterance_max_tokens]
        my_mask = torch.zeros(self.args.utterance_max_tokens, dtype=torch.long)
        my_mask[0] = 1
        input_encoding_mask.append(my_mask.unsqueeze(0))  # [1, utterance_max_tokens, utterance_max_tokens]


        #####################
        # response_gt = response + [self.eos] + [-100]
        response_gt = response + [self.eos] + [1]

        input_datas['encoding'] = torch.cat(input_encoding, dim=0)
        input_datas['segment_id'] = torch.cat(input_segment_id, dim=0)
        input_datas['encoding_mask'] = torch.cat(input_encoding_mask, dim=0)
        # print("input_datas['encoding'].shape=",input_datas['encoding'].shape)

        addition_nums = self.MAX_UTTERANCE_NUM - response_id - 1
        if addition_nums != 0:
            input_datas['encoding'] = torch.cat([input_datas['encoding'],
                                                 torch.zeros((addition_nums, self.args.utterance_max_tokens),
                                                             dtype=torch.long).fill_(self.pad)], dim=0)
            input_datas['segment_id'] = torch.cat([input_datas['segment_id'],
                                                   torch.zeros((addition_nums, self.args.utterance_max_tokens),
                                                               dtype=torch.long)], dim=0)
            input_datas['encoding_mask'] = torch.cat([input_datas['encoding_mask'],
                                                      torch.zeros((addition_nums, self.args.utterance_max_tokens),
                                                                  dtype=torch.long)], dim=0)
            # XXX: deal with NAN
            input_datas['encoding_mask'][response_id:, -1] = 1

        instance = {
            'graph': graph,
            'input_datas': input_datas,
            'decode_input': decode_input,
            'response': response_gt,
            'answer_index': response_id,
            'answer_from': example['answer_from']
        }

        return instance

    def __getitem__(self, index):
        example = self.examples[index]
        instance = self.build_graph_from_examples(
            example
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
        response = [ins['response'] for ins in batch]
        answer_index = [ins['answer_index'] for ins in batch]
        answer_from = [ins['answer_from'] for ins in batch]

        graph = dgl.batch(graph)
        encoding = torch.cat(encoding, dim=0)
        segment_id = torch.cat(segment_id, dim=0)
        encoding_mask = torch.cat(encoding_mask, dim=0)
        decode_input = torch.tensor(pad_ids(decode_input, self.pad))
        response = torch.tensor(pad_ids(response, self.pad))
        answer_index = torch.tensor(answer_index)
        answer_from = torch.tensor(answer_from)

        return graph, encoding, decode_input, segment_id, encoding_mask, response, answer_index, answer_from


class testDataset(BaseDataset):
    def __getitem__(self, index):
        example = self.examples[index]
        example['response'] = []
        instance = self.build_graph_from_examples(
            example,
            with_eos=False
        )
        return instance, example

    def collate_fn(self, batch):
        instance, examples = zip(*batch)
        graph, encoding, decode_input, segment_id, encoding_mask, response, answer_index, answer_from = super().collate_fn(
            instance)

        history = [example["history"] for example in examples]
        response = [example["response_text"] for example in examples]

        return graph, encoding, encoding_mask, answer_index, answer_from, (history, response)