import subprocess
from os.path import abspath, dirname, join

import click
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import (LearningRateMonitor, ModelCheckpoint,
                                         ModelSummary)

import kppr.datasets.datasets as datasets
import kppr.models.models as models


@click.command()
# Add your options here
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)',
              default=join(dirname(abspath(__file__)), 'config/config.yaml'))
@click.option('--data_config',
              '-dc',
              type=str,
              help='path to the config file (.yaml)',
              default=join(dirname(abspath(__file__)), 'config/oxford_data.yaml'))
@click.option('--weights',
              '-w',
              type=str,
              help='path to pretrained weights (.ckpt). Use this flag if you just want to load the weights from the checkpoint file without resuming training.',
              default=None)
@click.option('--checkpoint',
              '-ckpt',
              type=str,
              help='path to checkpoint file (.ckpt) to resume training.',
              default=None)
def main(config, data_config, weights, checkpoint):
    cfg = yaml.safe_load(open(config))
    data_cfg = yaml.safe_load(open(data_config))
    cfg['git_commit_version'] = str(subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD']).strip())
    cfg['data_config'] = data_cfg
    print(f"Start experiment {cfg['experiment']['id']}")
    # Load data and model
    data = datasets.getOxfordDataModule(data_cfg)

    model = models.getModel(
        cfg['network_architecture'], config=cfg,
        weights=weights)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_saver = ModelCheckpoint(monitor='val/recall_1',
                                       filename='best_{epoch:02d}',
                                       mode='max',
                                       save_last=True)

    tb_logger = pl_loggers.TensorBoardLogger('experiments/'+cfg['experiment']['id'],
                                             default_hp_metric=False)

    print('nr gpus:', cfg['train']['n_gpus'])
    # Setup trainer, Trainer(accelerator='gpu', devices=1) # v1.7+
    trainer = Trainer(gpus=cfg['train']['n_gpus'],
                      logger=tb_logger,
                      resume_from_checkpoint=checkpoint,
                      gradient_clip_val=0.2,
                      max_epochs=cfg['train']['max_epoch'],
                      callbacks=[lr_monitor, checkpoint_saver, ModelSummary(max_depth=2)],)

    # Train!
    trainer.fit(model, data)


if __name__ == "__main__":
    main()
