from torch.utils.data import DataLoader, RandomSampler,SequentialSampler
from transformers import get_linear_schedule_with_warmup

from torch.nn import CrossEntropyLoss

import torch
from torch.optim import AdamW

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression 

from tqdm import tqdm
import math

from ImageCausalModel import ImageCausalModel
from ImageDataLoader import CausalImageDataset

CUDA = (torch.cuda.device_count() > 0)


def platt_scale(outcome,probs):
    logits = logit(probs)
    logits = logits.reshape(-1,1)
    log_reg = LogisticRegression(penalty='none', warm_start = True, solver = 'lbfgs' )
    log_reg.fit(logits, outcome)
    return log_reg.predict_proba(logits)

def gelu(x):
    return 0.5 * x * (1.0 + torch.erf(x/math.sqrt(2.0)))

class CausalImageModelWrapper:
    def __init__(self, g_weight=1.0, Q_weight=0.1, batch_size=32):
        self.model = ImageCausalModel(num_labels=2, pretrained_model_names="resnet50")
        if CUDA:
            self.model = self.model.cuda()

        self.loss_weights = {
            'g': g_weight,
            'Q': Q_weight
        }
        self.batch_size = batch_size
        self.losses = []

    def train(self,images, confounds, treatments, outcomes , learning_rate = 2e-5, epochs  = 3):
        dataloader = self.build_dataloader(images, confounds, treatments, outcomes, batch_size = self.batch_size)
        self.model.train()
        optimizer = AdamW(self.model.parameters(), lr = learning_rate, eps = 1e-8)
        total_steps = len(dataloader) * epochs
        warmup_steps = total_steps * 0.1
        scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps = warmup_steps,num_training_steps = total_steps)

        for epoch in range(epochs):
            losses = []
            for batch in dataloader:
                if CUDA:
                    batch = tuple(x.cuda() for x in batch)
                images, confounds, treatments, outcomes = batch

                self.model.zero_grad()
                g_prob, Q_prob_T0, Q_prob_T1, g_logits, Q_logits_T0, Q_logits_T1 = self.model(images, confounds, treatments, outcomes)
                g_loss = CrossEntropyLoss()(g_logits, treatments)
                Q_loss_T0 = CrossEntropyLoss()(Q_logits_T0, outcomes)
                Q_loss_T1 = CrossEntropyLoss()(Q_logits_T1, outcomes)

                loss = self.loss_weights['g'] * g_loss + self.loss_weights['Q'] * (Q_loss_T0 + Q_loss_T1)

                loss.backward()
                optimizer.step()
                scheduler.step()
                losses.append(loss.detach().cpu().item())
            
        return self.model
    
    def inference(self, images, confounds, outcome = None):
        self.model.eval()
        dataloader = self.build_dataloader(images, confounds,outcomes = outcome,
                                           sampler = 'sequential')
        Q0s = []
        Q1s = []
        Ys = []
        for i, batch in tqdm(enumerate(dataloader),total = len(dataloader)):
            if CUDA: 
                batch = (x.cuda() for x in batch)
            images, confounds, _ ,outcomes = batch
            g_prob, Q0, Q1 ,_,_,_= self.model(images, confounds, outcome = outcomes)
            Q0s += Q0.detach().cpu().numpy().tolist()
            Q1s += Q1.detach().cpu().numpy().tolist()
            Ys += outcomes.detach().cpu().numpy().tolist()

            ## [todo] inferenceメソッドの形式?
        probs = np.array(list(zip(Q0s, Q1s)))
        preds = np.argmax(probs, axis = 1)    
        return probs, preds,Ys
    
    def ATE(self,C,image, Y = None, platt_scaling = False):
        ## [todo] ATEの計算方法
        Q_probs,_,Ys = self.inference(image,C,outcome = Y)
        if platt_scaling and Y is not None:
            Q0 = platt_scale(Ys, Q_probs[:,0])[:,0]
            Q1 = platt_scale(Ys, Q_probs[:,1])[:,1]
        else:
            Q0 = Q_probs[:,0]
            Q1 = Q_probs[:,1]
        return np.mean(Q0 - Q1)

    def build_dataloader(self,image_paths, confounds, treatments = None, outcomes = None,batch_size = 32,sampler = "random"):
        dataset = CausalImageDataset(image_paths, confounds, treatments, outcomes)
        sampler = RandomSampler(dataset) if sampler == "random" is not None else SequentialSampler(dataset)
        dataloader = DataLoader(dataset, batch_size = batch_size,sampler = sampler)
        return dataloader
    
if __name__ == '__main__':
    df = pd.read_csv("../input/outputs_v2.csv")
    print(df["no_of_ratings"].shape)
    ci = CausalImageModelWrapper(batch_size = 32, g_weight=0.1, Q_weight=0.1)
    ci.train(df["img_path"],df["no_of_ratings"], df["price_ave"], df["output_2v"],epochs = 32)
    print(ci.ATE(df["price_ave"], df["img_path"], platt_scaling = False))