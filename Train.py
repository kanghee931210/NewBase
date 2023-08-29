from modules.earlystoppers import EarlyStopper
from modules.recorders import Recorder
from modules.optimizers import get_optimizer
from modules.metrics import get_metric
from modules.losses import get_loss
from modules.trainer import Trainer

from model.utils import get_model

from modules.utils import load_yaml, get_logger, save_yaml
from modules.dataset import ETRIDataset_emo, ETRIDataset_emo_val

from datetime import datetime, timezone, timedelta
from torch.utils.data import DataLoader

import pandas as pd

import torch
import numpy as np
import random
import os
import shutil
import copy
import warnings
import wandb
from types import SimpleNamespace
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer_v2, optimizer_kwargs

try:
    api_key = ''
    wandb.login(key=api_key)
    anony = None
except:
    anony = "must"
    print('If you want to use your W&B account, go to Add-ons -> Secrets and provide your W&B access token. Use the Label name as wandb_api. \nGet your W&B access token from here: https://wandb.ai/authorize')


PROJECT_DIR = os.path.dirname(__file__)

config_path = os.path.join(PROJECT_DIR, 'config', 'ETRI_train_config_v2.yaml')
config = load_yaml(config_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

kst = timezone(timedelta(hours=9))
train_serial = datetime.now(tz=kst).strftime("%Y%m%d_%H%M%S")

# Recorder Directory
if config['LOGGER']['debug']:
    RECORDER_DIR = os.path.join(PROJECT_DIR, 'results', 'train', 'debug')
    # remove the record directory if it exists even though directory not empty
    if os.path.exists(RECORDER_DIR): shutil.rmtree(RECORDER_DIR)
else:
    RECORDER_DIR = os.path.join(PROJECT_DIR, 'results', 'train', train_serial)

os.makedirs(RECORDER_DIR, exist_ok=True)

if config['LOGGER']['wandb']:
    run = wandb.init(project='ETRI TASK1',
                     name=train_serial,
                     config=config,)

DATA_DIR = config['DIRECTORY']['train']
VAL_DIR = config['DIRECTORY']['val']

# Seed
torch.manual_seed(config['TRAINER']['seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(config['TRAINER']['seed'])
random.seed(config['TRAINER']['seed'])

def train():
    '''
            Set Logger
        '''

    logger = get_logger(name='train', dir_=RECORDER_DIR, stream=False)
    logger.info(f"Set Logger {RECORDER_DIR}")

    '''
           Load Data
    '''
    # Dataset

    df = pd.read_csv('../Dataset/info_etri20_emotion_train.csv')
    df_val = pd.read_csv('../Dataset/info_etri20_emotion_validation.csv')

    train_dataset = ETRIDataset_emo(df, base_path=DATA_DIR, img_size=config['DATASET']['img_size'])
    valid_dataset = ETRIDataset_emo_val(df_val, base_path=VAL_DIR, img_size=config['DATASET']['img_size'])

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=config['DATALOADER']['batch_size'],
                                  num_workers=config['DATALOADER']['num_workers'],
                                  shuffle=config['DATALOADER']['shuffle'],
                                  pin_memory=config['DATALOADER']['pin_memory'],
                                  drop_last=config['DATALOADER']['drop_last'])

    val_dataloader = DataLoader(dataset=valid_dataset,
                                batch_size=config['DATALOADER']['batch_size'],
                                num_workers=config['DATALOADER']['num_workers'],
                                shuffle=False,
                                pin_memory=config['DATALOADER']['pin_memory'],
                                drop_last=config['DATALOADER']['drop_last'])

    model_name = config['TRAINER']['model']
    model = get_model(model_name=model_name).to(device)

    # optimizer = get_optimizer(optimizer_name=config['OPT']['opt'])
    # optimizer = optimizer(params=model.parameters(), lr=config['OPT']['lr'],
    #                       weight_decay=config['OPT']['weight_decay'],momentum=config['OPT']['momentum'])

    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=SimpleNamespace(**config['OPT'])))
    lr_scheduler, num_epochs = create_scheduler(SimpleNamespace(**config['SCH']), optimizer)

    # Loss
    loss = get_loss(loss_name=config['TRAINER']['loss'])

    # Metric
    metrics = {metric_name: get_metric(metric_name) for metric_name in config['TRAINER']['metric']}

    early_stopper = EarlyStopper(patience=config['TRAINER']['early_stopping_patience'],
                                 mode=config['TRAINER']['early_stopping_mode'],
                                 logger=logger)

    # logger.info(f"Load data, train:{len(train_dataset)} val:{len(val_dataset)}")
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      lr_scheduler=lr_scheduler,
                      loss=loss,
                      metrics=metrics,
                      device=device,
                      logger=logger,
                      amp=None if config['TRAINER']['amp'] else None,
                      interval=config['LOGGER']['logging_interval'])

    recorder = Recorder(record_dir=RECORDER_DIR,
                        model=model,
                        optimizer=optimizer,
                        scheduler=None,
                        amp=None if config['TRAINER']['amp'] else None,
                        logger=logger)

    # Save train config
    save_yaml(os.path.join(RECORDER_DIR, 'train_config.yml'), config)

    n_epochs = config['TRAINER']['n_epochs']
    # num_epochs
    for epoch_index in range(num_epochs):

        # Set Recorder row
        row_dict = dict()
        row_dict['epoch_index'] = epoch_index
        row_dict['train_serial'] = train_serial
        """
        Train
        """
        print(f"Train {epoch_index}/{n_epochs}")
        logger.info(f"--Train {epoch_index}/{n_epochs}")

        trainer.train(dataloader=train_dataloader, epoch_index=epoch_index, mode='train')

        row_dict['train_loss'] = trainer.loss_mean
        row_dict['train_elapsed_time'] = trainer.elapsed_time
        row_dict['train_lr'] = trainer.lrs

        for metric_str, score in trainer.score_dict.items():
            # row_dict[f"train_{metric_str}_d"] = score[0]
            row_dict[f"train_acc_d"] = score[0]
            row_dict[f"train_acc_g"] = score[1]
            row_dict[f"train_acc_e"] = score[2]
            row_dict[f"train_acc"] = (score[0] + score[1] + score[2]) / 3

        wandb.log({"Loss": trainer.loss_mean})
        wandb.log({"D Acc": score[0]})
        wandb.log({"G Acc": score[1]})
        wandb.log({"E Acc": score[2]})
        wandb.log({"Acc": (score[0] + score[1] + score[2]) / 3})
        trainer.clear_history()

        """
        Validation
        """

        print(f"Val {epoch_index}/{n_epochs}")
        logger.info(f"--Val {epoch_index}/{n_epochs}")
        trainer.train(dataloader=val_dataloader, epoch_index=epoch_index, mode='val')

        row_dict['val_loss'] = trainer.loss_mean
        row_dict['val_elapsed_time'] = trainer.elapsed_time

        for metric_str, score in trainer.score_dict.items():
            row_dict[f"val_acc_d"] = score[0]
            row_dict[f"val_acc_g"] = score[1]
            row_dict[f"val_acc_e"] = score[2]
            row_dict[f"val_acc"] = (score[0] + score[1] + score[2]) / 3

        wandb.log({"Val Loss": trainer.loss_mean})
        wandb.log({"Val D Acc": score[0]})
        wandb.log({"Val G Acc": score[1]})
        wandb.log({"Val E Acc": score[2]})
        wandb.log({"Val Acc": (score[0] + score[1] + score[2]) / 3})
        trainer.clear_history()

        """
        Record
        """
        # Log results on the local
        recorder.add_row(row_dict)
        recorder.save_plot(config['LOGGER']['plot'])
        """
        Early stopper
        """
        early_stopping_target = config['TRAINER']['early_stopping_target']
        early_stopper.check_early_stopping(loss=row_dict[early_stopping_target])

        if (early_stopper.patience_counter == 0) or (epoch_index == n_epochs - 1):
            recorder.save_weight(epoch=epoch_index)
            best_row_dict = copy.deepcopy(row_dict)

        if early_stopper.stop == True:
            logger.info(
                f"Eearly stopped, counter {early_stopper.patience_counter}/{config['TRAINER']['early_stopping_patience']}")
            break


if __name__ == '__main__':
    train()
