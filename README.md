# Pytorch template
mlflow + hydra

## Setup
```bash
pip install -r requirements.txt
```

## Commands
```bash
python train.py  # 学習

python train.py resume=outpu/train/path/to/checkpoint.pth  # チェックポイントから再開

python train.py -m 'optimizer.lr=choice(0.1, 0.001, 0.0001)  # optuna

mlflow ui  # mlflow
tensorboard --logdir outputs/train/  # tensorboard

make format  # auto format
```

## 参考
- https://ymym3412.hatenablog.com/entry/2020/02/09/034644
- https://github.com/victoresque/pytorch-template