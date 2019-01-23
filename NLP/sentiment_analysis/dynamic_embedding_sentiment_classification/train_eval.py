from models import WordCNN
from sentiment_data_loader import create_data_loader

from argparse import ArgumentParser
from tqdm import tqdm
from torch.nn import DataParallel
from torch.optim import SGD

import torch
import torch.nn as nn
import mlflow

class Sentiment_train_eval(object):
    def __init__(self, train_loader, val_loader, optimizer, loss_fn, config):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.config = config

        #turn on tracker
        mlflow.start_run()

        for attr, value in self.config.items():
            mlflow.log_param(attr, value)

    def train_eval(self):
        for epoch in tqdm(range(self.config['epochs'])):
            for idx, batch in enumerate(tqdm(self.train_loader)):
                text, label = batch.document, batch.label
                text.data.t_()

                prediction = self.model(text)

                loss = self.loss_fn(prediction, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                mlflow.log_metric('train_loss', loss.item())

            n_total_data = 0.0
            n_correct = 0.0

            for idx, batch in enumerate(tqdm(self.val_loader)):
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

if __name__ == '__main__':
    parser = ArgumentParser()
    #data loader param
    parser.add_argument('--train_file_path', default='../data/naver_movie_sentiment/ratings_train.txt', help='training file path')
    parser.add_argument('--val_file_path', default='../data/naver_movie_sentiment/ratings_test.txt', help='validation file path')
 
    #model param
    parser.add_argument('--hidden_size', type=int, default=224)         #for filter setting
    parser.add_argument('--dim', type=int, default=224)                 #for embedding setting
    parser.add_argument('--n_channel_per_window', type=int, default=3)
    parser.add_argument('--label_size', type=int, default=2)
    parser.add_argument('--dropout', type=float, default='0.5')
 
    #experiment param
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum (default: 0.5)')

    config = parser.parse_args()

    #data loader ready
    train_loader, vocab_size = create_data_loader(config.train_file_path, build_vocab=True)
    val_loader = create_data_loader(config.val_file_path)

    #model ready
    model = WordCNN(vocab_size, config)
 
    optimizer = SGD(model.parameters(), lr=config.lr, momentum=config.momentum)
    loss_fn = nn.CrossEntropyLoss()

    trainer = Sentiment_train_eval(train_loader, val_loader, optimizer, loss_fn, vars(config))
    trainer.train_eval()
