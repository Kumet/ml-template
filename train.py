import logging
import os
import mlflow
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from hydra import utils
from src.logger import MlflowWriter
from src.trainer import Trainer
from src.utils import instantiate

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

logger = logging.getLogger('train')


@hydra.main(config_path='conf/', config_name='train')
def main(config: DictConfig):
    # setup data_loader instances
    data_loader, valid_data_loader = instantiate(config.data_loader)

    # build model. print it's structure and # trainable params.
    model = instantiate(config.arch)
    logger.info(model)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    logger.info(f'Trainable parameters: {sum([p.numel() for p in trainable_params])}')

    # get function handles of loss and metrics
    criterion = instantiate(config.loss, is_func=True)
    metrics = [instantiate(met, is_func=True) for met in config['metrics']]

    # build optimizer, learning rate scheduler.
    optimizer = instantiate(config.optimizer, model.parameters())
    lr_scheduler = instantiate(config.lr_scheduler, optimizer)

    # setup mlflow
    mlflow.set_tracking_uri('file://' + utils.get_original_cwd() + '/mlruns')
    mlflow_writer = MlflowWriter(config.experiment_name)
    mlflow_writer.log_params_from_omegaconf_dict(config)
    mlflow_writer.log_torch_model(model)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      mlflow_writer=mlflow_writer,
                      lr_scheduler=lr_scheduler)
    trainer.train()

    mlflow_writer.log_artifact(os.path.join(os.getcwd(), '.hydra/config.yaml'))
    mlflow_writer.log_artifact(os.path.join(os.getcwd(), '.hydra/hydra.yaml'))
    mlflow_writer.log_artifact(os.path.join(os.getcwd(), '.hydra/overrides.yaml'))
    mlflow_writer.log_artifact(os.path.join(os.getcwd(), 'train.log'))
    mlflow_writer.set_terminated()


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
