import hydra
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from hydra import utils
from omegaconf import DictConfig, ListConfig

from src.data_loader import get_data_loaders
from src.models.model import Net


def log_params_from_omegaconf_dict(params):
    for param_name, element in params.items():
        _explore_recursive(param_name, element)


def _explore_recursive(parent_name, element):
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                _explore_recursive(f'{parent_name}.{k}', v)
            else:
                mlflow.log_param(f'{parent_name}.{k}', v)
    elif isinstance(element, ListConfig):
        for i, v in enumerate(element):
            mlflow.log_param(f'{parent_name}.{i}', v)


@hydra.main(config_path='conf/', config_name='config')
def main(cfg):
    train_loader, test_loader = get_data_loaders('data/', cfg.train.batch_size)
    model = Net(cfg)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg.optimizer.lr,
                          momentum=cfg.optimizer.momentum)
    mlflow.set_tracking_uri('file://' + utils.get_original_cwd() + 'mlruns')
    mlflow.set_experiment(cfg.mlflow.runname)
    with mlflow.start_run():
        for epoch in range(cfg.train.epoch):
            running_loss = 0.0
            log_params_from_omegaconf_dict(cfg)
            for i, (x, y) in enumerate(train_loader):
                steps = epoch * len(train_loader) + i
                optimizer.zero_grad()

                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                mlflow.log_metric('loss', loss.item(), step=steps)

            correct = 0
            total = 0
            with torch.no_grad():
                for (x, y) in test_loader:
                    outputs = model(x)
                    _, predicted = torch.max(outputs.data, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()

            accuracy = float(correct / total)
            mlflow.log_metric('acc', accuracy, step=epoch)



if __name__ == '__main__':
    main()
