import torch
import torch.nn.functional as F
import numpy as np
from numpy import random
import random

def set_seed(args):
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if args.n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)

def average_distributed_scalar(scalar, args):
  """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
  if args.local_rank == -1:
    return scalar
  scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
  torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
  return scalar_t.item()

def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
  """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
      Args:
          logits: logits distribution shape (..., vocabulary size)
          top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
          top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
              whose total probability mass is greater than or equal to the threshold top_p.
              In practice, we select the highest probability tokens whose cumulative probability mass exceeds
              the threshold top_p.
          threshold: a minimal threshold to keep logits
  """
  top_k = min(top_k, logits.size(-1))
  if top_k > 0:
    # Remove all tokens with a probability less than the last token in the top-k tokens
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits[indices_to_remove] = filter_value

  if top_p > 0.0:
    # Compute cumulative probabilities of sorted tokens
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probabilities > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # Back to unsorted indices and set them to -infinity
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = filter_value

  indices_to_remove = logits < threshold
  logits[indices_to_remove] = filter_value

  return logits

