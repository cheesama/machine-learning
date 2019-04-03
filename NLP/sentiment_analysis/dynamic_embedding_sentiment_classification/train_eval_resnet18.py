from sentiment_data_loader import create_data_loader
from models import Word_Resnet

from argparse import ArgumentParser
from tqdm import tqdm
from torch.nn import DataParallel
from torch.optim import SGD
from torchvision import models
from bayes_opt import BayesianOptimization

import torch
import torch.nn as nn
import mlflow

class Sentiment_train_eval(object):
    def __init__(self, train_loader, val_loader, model, loss_fn, config):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.loss_fn = loss_fn
        self.config = config

    def bayes_opt_train_eval(self):
        pbounds = {}
        for attr, value in vars(self.config).items():
            if isinstance(value, tuple):
                pbounds[attr] = value
                #self.config.pop(attr, None)

        bayes_optimizer = BayesianOptimization(f=self.bayes_opt_config_wrapper, pbounds=pbounds, random_state=88) 
        bayes_optimizer.maximize(init_points=config.init_points, n_iter=config.n_iter,)

    def bayes_opt_config_wrapper(self, **params):
        for attr, value in params.items():
            setattr(self.config, attr, value)

        return self.train_eval()

    def train_eval(self):
        #turn on tracker
        mlflow.start_run()

        #model creation
        self.model = self.model(config)
        self.model.to(device)
        self.model = nn.DataParallel(self.model)

        optimizer = config.optimizer(self.model.parameters(), lr=self.config.lr, momentum=self.config.momentum)
        
        for attr, value in vars(self.config).items():
            if attr == 'resnet_model':
                continue

            mlflow.log_param(attr, value)

        for epoch in (range(self.config.epochs)):
            self.model.train()
            #print ('train epoch: ', str(epoch))
            for idx, batch in enumerate(self.train_loader):
                text, label = batch.document, batch.label
                
                text.data.t_()

                prediction = self.model(text)

                loss = self.loss_fn(prediction, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                mlflow.log_metric('train_loss', loss.item())

            n_total_data = 0.0
            n_correct = 0.0

            self.model.eval()
            #print ('eval epoch: ', str(epoch))
            for idx, batch in enumerate(self.val_loader):
                text, label = batch.document, batch.label

                text.data.t_()

                with torch.no_grad():
                    prediction = self.model(text)

                 # Calculate accuracy
                n_total_data += len(label)
                _, prediction = prediction.max(1)
                n_correct += (prediction == label).sum().item()

            accuracy = n_correct / n_total_data

            mlflow.log_metric('acc', accuracy)

        mlflow.end_run()

        return accuracy

if __name__ == '__main__':
    parser = ArgumentParser()
    #data loader param
    parser.add_argument('--train_file_path', default='../data/naver_movie_sentiment/ratings_train.txt', help='training file path')
    parser.add_argument('--val_file_path', default='../data/naver_movie_sentiment/ratings_test.txt', help='validation file path')
    parser.add_argument('--batch_size', type=int, default=512)          #based on 12GB GPU Mem

    #model param
    parser.add_argument('--hidden_size', type=int, default=224)         #for filter setting
    parser.add_argument('--dim', type=int, default=224)                 #for embedding setting(considering resnet input shape)
    parser.add_argument('--n_channel_per_window', type=int, default=30)
    parser.add_argument('--label_size', type=int, default=2)
    parser.add_argument('--dropout', type=float, default='0.1')
 
    #experiment param
    parser.add_argument('--epochs', type=int, default=40, help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', default=(0.005, 0.01), help='learning rate (default: 0.005)')
    parser.add_argument('--momentum', default=(0.5, 0.9), help='SGD momentum (default: 0.5)')
    parser.add_argument('--optimizer', default=SGD)

    #bayesian optimization setting
    parser.add_argument('--init_points', type=int, default=10)
    parser.add_argument('--n_iter', type=int, default=20)

    config = parser.parse_args()

    #set GPU device
    device = torch.device(0)

    #data loader ready(it takes long time because of building vocab)
    train_loader, val_loader, vocab_size = create_data_loader(config.train_file_path, config.val_file_path, batchSize=config.batch_size, device=device)

    #model ready
    config.vocab_size = vocab_size
    model = Word_Resnet
    config.resnet_model = models.resnet18()

    loss_fn = nn.CrossEntropyLoss()

    trainer = Sentiment_train_eval(train_loader, val_loader, model, loss_fn, config)
    #trainer.train_eval()
    trainer.bayes_opt_train_eval()
