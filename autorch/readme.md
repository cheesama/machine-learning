# Autorch

Autorch is an open source library for automated machine learning hyper parameter tuning using PyTorch. The ultimate goal of this tool is to provide convenient logging & status function about configuration which user set hyper parameters.

## Requirements

Ray, MLflow, python >= 3.6, pytorch==0.4.0, keras, tensorflow

## Workflow

![Overall Workflow](workflow.png)

## How to use

#### 1. Component implementation

In Autorch, user has to implement their own data loader, loss function and model architecture. But It can be easily found in many examples such as MNIST or Cifar10 example in this repository.

```python
#after install autorch
from autorch import train_eval
tuning = train_eval.Tuning()
```
After tuning wrapper call, compoenents can be set like below

| Components        | API                                              |
| ------------------ | ------------------------------------------------------------- |
| Model architecture | tuning.setCustomModel(model)                                 |
| Loss function      | tuning.setCriterion(criterion)                                |
| Data loader        | tuning.setCustomDataLoader(dataLoader)                        |
| Metric             | tuning.setCustomMetric(customMetric)                          |

#### 2. Setting experiment configuration

##### 2-1. Applying custom metric to test_epoch module

After these 4 components are implemented(or re-use other one), user can set experiment config like below

```python
#set the learning config
tuning.setConfigFile('cifar10','config.ini')
```

##### 2-2. Setting configuration to config.ini file

Here is Instruction of variety of variables in **config.ini**. Please refer example file.(based on example)

| Variable Name            | Role                                                         |
| ------------------------ | ------------------------------------------------------------ |
| [cifar10]                | Configuration sectio name(it can be re-name by user, if there are several configuration in config.ini file, it can be splitter by this section name) |
| model_save_interval(int) | Set the model binary file saving period. It depends on 'epoch' |
| multiGPU('Y' or 'N')     | If user set multiGPU as 'Y', the model is built as 'DataParallel' mode. But when model will be saved just 1 GPU model. |
| epoch(int)               | Set the experiment maximum epoch                             |
| batch_size(int)          | Set the batch size of model                                  |
| train_log_interval       | Set the training phase log interval period based on multi batch |
| optimizer(string)        | Set the optimizer type(currently it support -> [sgd, rmsprop, adadelta, agagard, sparseAdam, adamax, asgd, lbfgs,  rpop, adam(default)) |
| learning_rate            | Set ther learning rate of optimizer. It supports grid & random search mode. |
| momentum                 | Set ther momentum of optimizer. It supports grid & random search mode. |
| mlflow_tracking_URI      | Set ther MLFlow ui server URI. user can check total experiment result easily via accessing this URI |
| ray_dir                  | Set hyper parameter tuning result saving path                |
| model_save_dir           | Set saved model path. It will be created in ray_dir          |
| model_name_prefix        | Set saved model name prefix. Model name will be prefix - epoch form. |
| num_samples              | Set the tuning trial number. If it set with grid search parameters, n times grid search tuning experiment will be done. |
| trial_resources_cpu      | Set assigning CPU resource to each experiment. Each experiment can be started with other experiment. |
| trial_resources_gpu      | Set assigning GPU resource to each experiment. Each experiment can be started with other experiment. |

In some hyper parameters, user can set some list or range. If user set hyper parameters like **[0.1, 0.2, 0.3]**, it means **grid search**. With that hyper parameters, the experiment will be done via parameters grid search policy. In the other hands, if user set hyper parameters like **(0.1,0.5)**, it means uniform **random search**.

So user can apply these two policy to each hyper parameters.

#### 3. Checking experiment results

After tuning experiments are finished, user can check the result of experiement like this.

```bash
TERMINATED trials:
 - tune_train_eval_0_learning_rate=0.001,momentum=0.1:  TERMINATED [pid=27242], 88 s, 0 ts, 0.00461 loss, 10 acc
 - tune_train_eval_1_learning_rate=0.01,momentum=0.1:   TERMINATED [pid=27239], 88 s, 0 ts, 0.00417 loss, 24.8 acc
 - tune_train_eval_2_learning_rate=0.001,momentum=0.5:  TERMINATED [pid=27244], 90 s, 0 ts, 0.0046 loss, 10.5 acc
 - tune_train_eval_3_learning_rate=0.01,momentum=0.5:   TERMINATED [pid=27243], 90 s, 0 ts, 0.0037 loss, 33.4 acc
```

And via accessing MLflow ui server written in config.ini, user can compare among tuned experiment more specifically.

## To Do
1. Adding more optimizer options to config.ini to support various mode
2. Supporting convert function from pytroch model to keras(tesorflow) model
3. visualization function support

