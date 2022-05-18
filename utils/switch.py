import dataloaders
import helpers
# import models
import importlib

def import_from_path(module_name, module_file_path):
  module_spec = importlib.util.spec_from_file_location(module_name, module_file_path)
  module = importlib.util.module_from_spec(module_spec)
  module_spec.loader.exec_module(module)
  return module


def get_modules(args):
  config = args

  dataloader = import_from_path("dataloader", f"dataloaders/{config.dataloader}.py")
  helper = import_from_path("helper", f"helpers/{config.helper}.py")
  model = import_from_path("model", f"models/{config.model}.py")

  output = (dataloader, model, helper)
  return output

def get_train_aux(args, model):
  optimizers = helpers.optimizer(args, model)
  optimizer = getattr(optimizers, args.optimizer)

  lr_schedulers = helpers.lr_scheduler(args, optimizer)
  lr_scheduler = getattr(lr_schedulers, args.lr_scheduler)

  return optimizer, lr_scheduler