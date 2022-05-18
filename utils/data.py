import os
import re
import json
import logging

logger = logging.getLogger(__name__)

RE_ART = re.compile(r'\b(a|an|the)\b')
RE_PUNC = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')

def add_special_tokens_(model, tokenizer, ATTR_TO_SPECIAL_TOKEN):
  # ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
  #                          'additional_special_tokens': ['<speaker1>', '<speaker2>']}
  """ Add special tokens to the tokenizer and the model if they have not already been added. """
  orig_num_tokens = len(tokenizer.encoder)
  num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)  # doesn't add if they are already there
  if num_added_tokens > 0:
    model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)


def remove_articles(_text):
  return RE_ART.sub(' ', _text)


def white_space_fix(_text):
  return ' '.join(_text.split())


def remove_punc(_text):
  return RE_PUNC.sub(' ', _text)  # convert punctuation to spaces


def lower(_text):
  return _text.lower()


def normalize(text):
  """Lower text and remove punctuation, articles and extra whitespace. """
  return white_space_fix(remove_articles(remove_punc(lower(text))))


def write_detection_preds(dataset_walker, output_file, data_infos, pred_ids):
  # Flatten the data_infos
  data_infos = [
    {"dialog_id": info["dialog_ids"][i]}
    for info in data_infos
    for i in range(len(info["dialog_ids"]))
  ]

  labels = [{"target": False}] * len(dataset_walker)
  # Update the dialogs with detection result
  for info, pred_id in zip(data_infos, pred_ids):
    dialog_id = info["dialog_id"]
    label = {"target": bool(pred_id)}
    labels[dialog_id] = label

  if os.path.dirname(output_file) and not os.path.exists(os.path.dirname(output_file)):
    os.makedirs(os.path.dirname(output_file))

  with open(output_file, "w") as jsonfile:
    logger.info("Writing predictions to {}".format(output_file))
    json.dump(labels, jsonfile, indent=2)


def write_selection_preds(dataset_walker, output_file, data_infos, sorted_pred_ids, topk=5):
  # Flatten the data_infos
  data_infos = [
    {
      "dialog_id": info["dialog_ids"][i],
      "candidate_keys": info["candidate_keys"][i]
    }
    for info in data_infos
    for i in range(len(info["dialog_ids"]))
  ]

  labels = [label for log, label in dataset_walker]
  new_labels = [{"target": False}] * len(dataset_walker)
  # Update the dialogs with selected knowledge
  for info, sorted_pred_id in zip(data_infos, sorted_pred_ids):
    dialog_id = info["dialog_id"]
    candidate_keys = info["candidate_keys"]

    snippets = []
    for pred_id in sorted_pred_id[:topk]:
      selected_cand = candidate_keys[pred_id]
      domain, entity_id, doc_id = selected_cand.split("__")
      snippet = {
        "domain": domain,
        "entity_id": "*" if entity_id == "*" else int(entity_id),
        "doc_id": int(doc_id)
      }
      snippets.append(snippet)

    new_label = {"target": True, "knowledge": snippets}
    label = labels[dialog_id]
    if label is None:
      label = new_label
    else:
      label = label.copy()
      if "response_tokenized" in label:
        label.pop("response_tokenized")
      label.update(new_label)

    new_labels[dialog_id] = label

  if os.path.dirname(output_file) and not os.path.exists(os.path.dirname(output_file)):
    os.makedirs(os.path.dirname(output_file))

  with open(output_file, "w") as jsonfile:
    logger.info("Writing predictions to {}".format(output_file))
    json.dump(new_labels, jsonfile, indent=2)


def write_generation_preds(dataset_walker, output_file, dialog_ids, responses):
  labels = [label for log, label in dataset_walker]
  new_labels = [{"target": False}] * len(dataset_walker)
  # Update the dialogs with detection result
  for dialog_id, response in zip(dialog_ids, responses):
    label = labels[dialog_id]
    new_label = {"target": True, "response": response}
    if label is None:
      label = new_label
    else:
      label = label.copy()
      label.update(new_label)
      if "response_tokenized" in label:
        label.pop("response_tokenized")
    new_labels[dialog_id] = label

  if os.path.dirname(output_file) and not os.path.exists(os.path.dirname(output_file)):
    os.makedirs(os.path.dirname(output_file))

  with open(output_file, "w") as jsonfile:
    logger.info("Writing predictions to {}".format(output_file))
    json.dump(new_labels, jsonfile, indent=2)

def truncate_sequences(sequences, max_length):
  words_to_cut = sum(list(map(len, sequences))) - max_length
  if words_to_cut <= 0:
    return sequences

  while words_to_cut > len(sequences[0]):
    words_to_cut -= len(sequences[0])
    sequences = sequences[1:]

  sequences[0] = sequences[0][words_to_cut:]
  return sequences

def truncate_utterance(sequences, max_length):
  for index, utterance in enumerate(sequences):
    sequences[index] = utterance[:max_length]
    
  return sequences

def pad_ids(arrays, padding, max_length=-1):
  if max_length < 0:
    max_length = max(list(map(len, arrays)))

  arrays = [
    array + [padding] * (max_length - len(array))
    for array in arrays
  ]

  return arrays

def pad_ids_2D(arrays, padding, max_length=-1):
  if max_length < 0:
    max_length = max(list(map(len, arrays)))

  arrays = [
    [row + [padding] * (max_length - len(row)) for row in array]
    for array in arrays
  ]

  arrays = [
    array + [[padding] * max_length for _ in range(max_length - len(array))]
    for array in arrays
  ]

  return arrays