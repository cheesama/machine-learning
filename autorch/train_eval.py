import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import os, sys, time
import configparser, multiprocessing
import numpy as np

import mlflow           #for logging experiment result

import ray
from ray import tune    #for tuning hyper-parameters
from ray.tune.schedulers import AsyncHyperBandScheduler

#set custom model & data_loader classes
from model import cifar10_classification_model
from loader import cifar10_image_loader

#set the learning config
config = configparser.ConfigParser()
config.read('config.ini')
config = config['cifar10'] #section config load

#set the model
model = cifar10_classification_model.Cifar10_classifier()

#set the loss function(if you implement your own, import that custom loss class)
criterion = nn.CrossEntropyLoss()

#set the dataLoader
dataLoader = cifar10_image_loader.Cifar10ImageLoader(data_dir=config['data_dir'], batch_size=int(config['batch_size']))

def tune_train_eval(loader, model, criterion, config, tuned, reporter):
    for key, value in tuned.items():
        config[key] = str(value)

    #choose model type(whether DataParallel or not)
    if 'multiGPU' in config.keys() and config['multiGPU'] == 'Y':
        model = nn.DataParallel(model)

    if torch.cuda.is_available():
        model = model.cuda()

    #default optimizer -> adam
    optimizer = optim.Adam(model.parameters())

    #start setting optimizer
    if 'optimizer' in config.keys():
        if config['optimizer'] =='sgd':
            optimizer = optim.SGD(model.parameters(), lr=float(config['learning_rate']))
        elif config['optimizer'] == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters())
        elif config['optimizer'] == 'adadelta':
            optimizer = optim.Adadelta(model.parameters())
        elif config['optimizer'] == 'adagrad':
            optimizer = optim.Adagrad(model.parameters())
        elif config['optimizer'] == 'sparseAdam':
            optimizer = optim.SparseAdam(model.parameters())
        elif config['optimizer'] == 'adamax':
            optimier = optim.Adamax(model.parameters())
        elif config['optimizer'] == 'asgd':
            optimizer = optim.ASGD(model.parameters())
        elif config['optimizer'] == 'lbfgs':
            optimizer = optim.LBFGS(model.parameters())
        elif config['optimizer'] == 'rprop':
            optimizer = optim.Rprop(model.parameters())

    optimizer.param_groups[0]['lr'] = float(config['learning_rate'])

    if 'momentum' in config.keys():
        optimizer.param_groups[0]['momentum'] = float(config['momentum'])
    if 'weight_deacy' in config.keys():
        optimizer.param_groups[0]['weight_decay'] = float(config['weight_decay'])
    if 'nesterov' in config.keys():
        optimizer.param_groups[0]['nesterov'] = float(config['nesterov'])
    #end setting optimizer

    #prepare model save dir
    if 'model_save_dir' in config.keys():
        if not os.path.isdir(config['model_save_dir']):
            os.mkdir(config['model_save_dir'])

    #prepare trainLoder, testLoder seperately
    trainLoader = dataLoader.trainLoader
    testLoader = dataLoader.testLoader

    def train_epoch(epoch, loader, model, criterion, config, optimizer):
        model.train()
        for batch_idx, (data, target) in enumerate(loader):
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if 'train_log_interval' in config.keys():
                if batch_idx % int(config['train_log_interval']) == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(loader.dataset), 100. * batch_idx / len(loader), loss.data.item()))
                    if 'mlflow_tracking_URI' in config.keys():
                        mlflow.log_metric('train_loss', loss.data.item())

    def test_epoch(loader, model, criterion, config):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(loader):
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)

                output = model(data)
                test_loss += criterion(output, target).sum().item() # sum up batch loss
                pred = output.data.max(1)[1]                        # get the index of the max log-probability
                correct += pred.eq(target.data).cpu().sum().item()  # classification task prediction & correction

        test_loss /= len(loader.dataset)
        test_accuracy = 100.0 * correct / len(loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(loader.dataset), test_accuracy))

        if 'mlflow_tracking_URI' in config.keys():
            mlflow.log_metric('test_loss', test_loss)
            mlflow.log_metric('test_accuracy', test_accuracy)

        print ('test acc:' + str(test_accuracy))
        reporter(mean_loss=test_loss, mean_accuracy=test_accuracy)

    #set MLflow tracking server
    if 'mlflow_tracking_URI' in config.keys():
        print ("MLflow Tracking URI: %s" % (config['mlflow_tracking_URI']))
        mlflow.set_tracking_uri(config['mlflow_tracking_URI'])

        with mlflow.start_run():
            print ('setting parameters')
            for key, value in config.items():
                mlflow.log_param(key, value)
                print (key + '\t:\t' + value)

            for epoch in range(1, int(config['epoch']) + 1):
                print ('epoch: ' + str(epoch))

                train_epoch(epoch, trainLoader, model, criterion, config, optimizer)
                if 'model_save_dir' in config.keys() and 'model_save_interval' in config.keys():
                    if epoch % int(config['model_save_interval']) == 0:
                        if 'multiGPU' in config.keys() and config['multiGPU'] == 'Y':                   
                            torch.save(model.module, os.getcwd() + os.sep + config['model_save_dir'] + os.sep + config['model_name_prefix'] + '_epoch_' + str(epoch) + '.pkl')
                        else:
                            torch.save(model, os.getcwd() + os.sep + config['model_save_dir'] + os.sep + config['model_name_prefix'] + '_epoch_' + str(epoch) + '.pkl')
                        print ('model saved: ' + os.getcwd() + os.sep + config['model_save_dir'] + os.sep + config['model_name_prefix'] + '_epoch_' + str(epoch) + '.pkl')

                test_epoch(testLoader, model, criterion, config)

if __name__ == '__main__':
    ray.init()
    sched = AsyncHyperBandScheduler(time_attr="training_iteration", reward_attr="neg_mean_loss", max_t=400, grace_period=20)
    tune.register_trainable("tune_train_eval", lambda tuned, rprtr: tune_train_eval(dataLoader, model, criterion, config, tuned, rprtr))

    if 'mlflow_tracking_URI' in config.keys():
        host = config['mlflow_tracking_URI'].split('//')[1].split(':')[0]
        port = config['mlflow_tracking_URI'].split('//')[1].split(':')[1]
        os.system('mlflow ui -h ' + host + ' -p ' + port + ' &')
        print ('mlflow server start')

    experiment_config = {}
    experiment_config['exp'] = {}

    experiment_config['exp']['trial_resources']={}

    if config['multiGPU'] == 'Y':
        experiment_config['exp']['trial_resources']['gpu'] = int(torch.cuda.device_count())
    else:
        experiment_config['exp']['trial_resources']['gpu'] = 1

    experiment_config['exp']['run'] = "tune_train_eval"
    experiment_config['exp']['stop'] = {}
    experiment_config['exp']['stop']['training_iteration'] = int(config['epoch'])

    print ('tuning experiment config')
    print (experiment_config)

    tune.run_experiments(
        {
            "exp": {
                "stop": {
                    "training_iteration": int(config['epoch']),
                },
                "trial_resources": {
                    "gpu": 0,
                    "cpu": 1,
                },
                "run": "tune_train_eval",
                "num_samples": 5,
                "config": {
                    "learning_rate": lambda spec: np.random.uniform(0.001, 0.1),
                    "momentum": lambda spec: np.random.uniform(0.1, 0.9),
                },
                "local_dir": config['ray_dir'],
            }
        },
        verbose=0,
        scheduler=sched)

