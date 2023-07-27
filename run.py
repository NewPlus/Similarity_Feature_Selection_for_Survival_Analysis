import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger

from Survival_CostFunc_CIndex import neg_par_log_likelihood, c_index

parser = argparse.ArgumentParser()
parser.add_argument('-data_path', help=' : Please set the save data path', default="./data/") 
parser.add_argument('-gpus', help=' : Please set the gpu number(input str like "0")')
parser.add_argument('-method', help=' : similarity(Similarity Feature Selection) or cox_en(Cox-EN Feature Selection)')
parser.add_argument('-epoch', type=int, help=' : Please set the epoch', default=1000)
parser.add_argument('-batch_size', type=int, help=' : Please set the batch size', default=256)
parser.add_argument('-dropout_rate', type=float, help=' : Please set the dropout rate', default=0.8) 
parser.add_argument('-act_func', help=' : tanh(Hyperbolic Tan), sigm(Sigmoid), relu(ReLU), gelu(GELU), leak(Leaky ReLU)') 
args = parser.parse_args()

# 데이터셋 클래스 정의
class CancerSurvivalDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
        self.features = self.data.drop(['OS_EVENT', 'OS_MONTHS'], axis=1).values
        self.labels_status = self.data['OS_EVENT'].values
        self.labels_month = self.data['OS_MONTHS'].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        feature = torch.FloatTensor(self.features[index])
        label_status = torch.FloatTensor([self.labels_status[index]])
        label_month = torch.FloatTensor([self.labels_month[index]])
        return feature, label_status, label_month

# Cox-nnet 모델 정의
class CoxNnet(pl.LightningModule):
    def __init__(self, input_dim, dropout_rate=0.8, learning_rate=1e-3, actf="tanh"):
        super(CoxNnet, self).__init__()
        self.hidden_layer1 = nn.Linear(input_dim, 128)
        self.hidden_layer2 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        if actf == "tanh":
            self.activation_func = nn.Tanh()
        elif actf == "sigm":
            self.activation_func = nn.Sigmoid()
        elif actf == "relu":
            self.activation_func = nn.ReLU()
        elif actf == "gelu":
            self.activation_func = nn.GELU()
        elif actf == "leak":
            self.activation_func = nn.LeakyReLU()
        self.loss = neg_par_log_likelihood
        self.learning_rate = learning_rate

    def forward(self, x):
        x = self.activation_func(self.hidden_layer1(x))
        x = self.dropout(x)
        x = self.activation_func(self.hidden_layer2(x))
        x = self.dropout(x)
        output_status = self.output_layer(x)
        return output_status

    def training_step(self, batch, batch_idx):
        inputs, labels_status, labels_month = batch
        output_status = self.forward(inputs)
        loss = self.loss(output_status, labels_status, labels_month)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels_status, labels_month = batch
        output_status = self.forward(inputs)
        loss = self.loss(output_status, labels_status, labels_month)
        c_index_value = c_index(output_status, labels_status, labels_month)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("c_index", c_index_value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    
    def test_step(self, batch, batch_idx):
        inputs, labels_status, labels_month = batch
        output_status = self.forward(inputs)
        c_index_value = c_index(output_status, labels_status, labels_month)
        self.log("test_c_index", c_index_value, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)


if __name__ == '__main__' :
    argv = sys.argv
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpus  # Set the GPU 0 to use

    BATCH_SIZE = args.batch_size
    dropout_rate = args.dropout_rate
    lr = 5e-5
    actf = args.act_func # "tanh","sigm","relu","gelu","leak"
    # 1 -> "gelu"(but weird learning curb)
    # 2 -> "relu"(but low c-index)
    # 3 -> "tanh"(but low c-index)

    wandb.init(name="layer=2,similarity,lr=tune, drop=0.8, tanh", project='UNLV_group3_project4_after_features')
    wandb_logger = WandbLogger()

    # 데이터 로드
    train_data_path = args.data_path+"train_"+args.method+"_result.csv" # "./data/"
    valid_data_path = args.data_path+"valid_"+args.method+"_result.csv"
    test_data_path = args.data_path+"test_"+args.method+"_result.csv"

    train_data = pd.read_csv(train_data_path)
    valid_data = pd.read_csv(valid_data_path)
    test_data = pd.read_csv(test_data_path)
    train_data.sort_values("OS_MONTHS", ascending = False, inplace = True)

    train_dataset = CancerSurvivalDataset(train_data)
    val_dataset = CancerSurvivalDataset(valid_data)
    test_dataset = CancerSurvivalDataset(test_data)

    # 데이터 로더 설정
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=20)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=20)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=20)

    # Cox-nnet 모델 인스턴스화
    input_dim = train_data.shape[1] - 2  # 입력 차원 설정 (label 열 제외)
    model = CoxNnet(input_dim, dropout_rate=dropout_rate, actf=actf, learning_rate=lr)

    # Trainer 초기화 및 학습 실행
    trainer = pl.Trainer(gpus='1', max_epochs=args.epoch, logger=wandb_logger, auto_lr_find='learning_rate')
    lr_finder = trainer.tuner.lr_find(model, train_loader, val_loader, min_lr=1e-08, max_lr=1e-3, num_training=100)
    model.hparams.learning_rate = lr_finder.suggestion()

    trainer.fit(model, train_loader, val_loader)
    # Trainer를 사용하여 모델 테스트
    trainer.test(dataloaders=test_loader,ckpt_path='best')
