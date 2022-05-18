from transformers import AdamW
from ignite.contrib.handlers import PiecewiseLinear, ParamGroupScheduler


class optimizer(object):
  def __init__(self, args, model) -> None:
    self.args = args
    self.model = model
    
  @property
  def AdamW(self):
    return AdamW(self.model.parameters(), lr=self.args.lr)

  @property
  def split_decode(self):
    all_params = self.model.parameters()
    lm_head_params = list(self.model.lm_head.parameters())
    params_id = list(map(id, lm_head_params))
    other_params = list(filter(lambda p: id(p) not in params_id, all_params))
    opt = AdamW([
        {'params': other_params},
        {'params': lm_head_params, 'lr': self.args.lr / 5.0}
    ], lr=self.args.lr)
    return opt


class lr_scheduler(object):
  def __init__(self, args, optimizer) -> None:
    self.args = args
    self.optimizer = optimizer
  
  @property
  def one_step(self):
    args = self.args
    optimizer = self.optimizer
    steps_per_epoch = args.steps_per_epoch
    t_total = steps_per_epoch * args.n_epochs
    lr_scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (t_total, 0.0)])

    return lr_scheduler
  
  @property
  def split_decode(self):
    args = self.args
    optimizer = self.optimizer
    steps_per_epoch = args.steps_per_epoch
    t_total = steps_per_epoch * args.n_epochs
    scheduler1 = PiecewiseLinear(optimizer, 'lr', [(0, 0.0), (steps_per_epoch, args.lr), (t_total, 0.0)], param_group_index=0)
    scheduler2 = PiecewiseLinear(optimizer, 'lr', [(0, 0.0), (steps_per_epoch * 2, args.lr / 5.0), 
                                                   (steps_per_epoch * 2 + 1, args.lr), (t_total, 0.0)], param_group_index=1)
    lr_schedulers = [scheduler1, scheduler2]
    names = ["lr (base)", "lr (decoder)"]
    scheduler = ParamGroupScheduler(schedulers=lr_schedulers, names=names)
    return scheduler