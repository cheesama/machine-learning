from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from torchvision import datasets, models, transforms

import torch
import torch.nn as nn
import torch.nn.functional as F

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

from tensorboardX import SummaryWriter

from argparse import ArgumentParser

from SOP_DataLoader import SOPDataset
from SOP_Losses import MultiBatch_Contrastive_Loss
from SOP_Models import SOP_resnet18_embedding

def create_summary_writer(model, data_loader, log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    data_loader_iter = iter(data_loader)
    x, y = next(data_loader_iter)

    try:
        if model.module:
            writer.add_graph(model.module, x)
        else:
            writer.add_graph(model, x)
    except Exception as e:
        print("Failed to save model graph: {}".format(e))
    
    return writer

def run(targetFolder, trainFile, valFile, train_transform, val_transform, model, epochs, lr, momentum, log_interval, log_dir, batchSize=256):
    train_dataset = SOPDataset(targetFolder, trainFile, transform=train_transform)
    val_dataset = SOPDataset(targetFolder, valFile, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=False)
    val_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=False)
    
    writer = create_summary_writer(model, train_loader, log_dir)
    device = 'cpu'

    if torch.cuda.is_available():
        device = 'cuda'

    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    optimizer = Adam(model.parameters(), lr=lr)
    
    criterion = MultiBatch_Contrastive_Loss()
    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
   
    '''
    evaluator = create_supervised_evaluator(model,
                                            metrics={
                                                        'accuracy': Accuracy(),
                                                        'contrastive': MultiBatch_Contrastive_Loss,
                                                    },
                                            device=device)
    '''

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1
        if iter % log_interval == 0:
            print("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}"
                  "".format(engine.state.epoch, iter, len(train_loader), engine.state.output))
            writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)
    
    #@trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(engine.state.epoch, avg_accuracy, avg_nll))
        writer.add_scalar("training/avg_loss", avg_nll, engine.state.epoch)
        writer.add_scalar("training/avg_accuracy", avg_accuracy, engine.state.epoch)

    #@trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(engine.state.epoch, avg_accuracy, avg_nll))
        writer.add_scalar("valdation/avg_loss", avg_nll, engine.state.epoch)
        writer.add_scalar("valdation/avg_accuracy", avg_accuracy, engine.state.epoch)

    # kick everything off
    trainer.run(train_loader, max_epochs=epochs)

    writer.close()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--folder', help='image data folder(including train & val txt files)')
    parser.add_argument('--train', help='image pair training txt file')
    parser.add_argument('--val', help='image pair val txt file')
    parser.add_argument('--batch_size', type=int, default=1024, help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum (default: 0.5)')
    parser.add_argument('--log_interval', type=int, default=1, help='how many batches to wait before logging training status')
    parser.add_argument("--log_dir", type=str, default="tensorboard_logs", help="log directory for Tensorboard log output")
    parser.add_argument("--model_parallel", type=bool, default=True, help="set model parallel mode")
    
    args = parser.parse_args()

    embeddingnet = models.resnet18(pretrained=True)
    model = SOP_resnet18_embedding(embeddingnet)

    if args.model_parallel:
        model = nn.DataParallel(model)

    train_transform = transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])

    val_transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])


    run(args.folder, args.train, args.val, train_transform, val_transform, model, args.epochs, args.lr, args.momentum, args.log_interval, args.log_dir, args.batch_size)
