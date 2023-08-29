import torch
from time import time
from tqdm import tqdm


class Trainer():

    def __init__(self, model, optimizer,lr_scheduler, loss, metrics, device, logger, amp,interval=100):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss = loss
        self.metrics = metrics
        self.device = device
        self.logger = logger
        self.amp = amp

        self.interval = interval

        # History
        self.loss_sum = 0  # Epoch loss sum
        self.loss_mean = 0  # epoch loss mean
        self.filenames = list()

        self.d = list()
        self.d_preds = list()

        self.g = list()
        self.g_preds = list()

        self.e = list()
        self.e_preds = list()

        self.lrs = 0

        self.score_dict = dict()
        self.elapsed_time = 0

    def train(self, mode, dataloader, epoch_index=0):
        start_timestamp = time()
        self.model.train() if mode == 'train' else self.model.eval()

        for batch_index, sample in enumerate(tqdm(dataloader)):
            for key in sample:
                sample[key] = sample[key].to(self.device)

            out_daily, out_gender, out_embel = self.model(sample['image'])
            # Loss
            loss_daily = self.loss(out_daily, sample['daily_label'])
            loss_gender = self.loss(out_gender, sample['gender_label'])
            loss_embel = self.loss(out_embel, sample['embel_label'])
            loss = loss_daily + loss_gender + loss_embel

            # Update
            if mode == 'train':
                self.optimizer.zero_grad()
                if self.amp is None:
                    loss.backward()
                else:
                    with self.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                self.optimizer.step()


            elif mode in ['val', 'test']:
                pass
            else:
                raise ValueError('Mode should be either train, val, or test')

            # History
            # self.filenames += filename
            self.loss_sum += loss.item()
            # y_pred = (y_pred > 0.5).to(torch.int)
            daily_pred = out_daily.argmax(1)
            gender_pred = out_gender.argmax(1)
            embel_pred = out_embel.argmax(1)

            self.d_preds.append(daily_pred)
            self.d.append(sample['daily_label'])

            self.g_preds.append(gender_pred)
            self.g.append(sample['gender_label'])

            self.e_preds.append(embel_pred)
            self.e.append(sample['embel_label'])



            # Logging
            if batch_index % self.interval == 0:
                msg = f"batch: {batch_index}/{len(dataloader)} loss: {loss.item()}"
                self.logger.info(msg)
        if mode == 'train':
            self.lr_scheduler.step(epoch_index)

        # Epoch history
        self.loss_mean = self.loss_sum / len(dataloader)
        self.lrs = self.optimizer.param_groups[0]["lr"]

        # Metric
        self.d_preds = torch.cat(self.d_preds, dim=0).cpu().tolist()
        self.d = torch.cat(self.d, dim=0).cpu().tolist()

        self.g_preds = torch.cat(self.g_preds, dim=0).cpu().tolist()
        self.g = torch.cat(self.g, dim=0).cpu().tolist()

        self.e_preds = torch.cat(self.e_preds, dim=0).cpu().tolist()
        self.e = torch.cat(self.e, dim=0).cpu().tolist()

        for metric_name, metric_func in self.metrics.items():
            d_score = metric_func(self.d, self.d_preds)
            g_score = metric_func(self.g, self.g_preds)
            e_score = metric_func(self.e, self.e_preds)

            self.score_dict[metric_name] = d_score,g_score,e_score

        # Elapsed time
        end_timestamp = time()
        self.elapsed_time = end_timestamp - start_timestamp

    def clear_history(self):
        self.loss_sum = 0
        self.loss_mean = 0
        self.d_preds = list()
        self.d = list()
        self.g_preds = list()
        self.g = list()
        self.e_preds = list()
        self.e = list()
        self.score_dict = dict()
        self.lrs = 0
        self.elapsed_time = 0

