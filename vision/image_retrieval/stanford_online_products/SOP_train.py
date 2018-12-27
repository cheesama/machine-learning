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
from sklearn.metrics import recall_score
from tqdm import tqdm

from SOP_DataLoader import SOPDataset
from SOP_Losses import MultiBatch_Contrastive_Loss
from SOP_Models import SOP_resnet18_embedding

import faiss
import numpy as np
import os, sys

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


def calculate_recall_score(model, loader, feature_dim=1024, K=1, show_embedding=False, logger=None):
    model.eval()

    image_features = torch.Tensor(0, feature_dim).cpu().numpy()
    labels = np.array([])

    #based on resnet18 model input
    image_collection = torch.Tensor(0, 3, 224,224).cpu()

    print ('Building Image features')
    with torch.no_grad():
        for images, batch_labels in tqdm(loader):
            batch_features = model(images)     
            image_features = np.concatenate((image_features, batch_features.cpu().numpy()), axis=0)

            labels = np.concatenate((labels, batch_labels), axis=0)

            if show_embedding:
                image_collection = torch.cat((image_collection, images))

    if show_embedding and logger:
        #for fast loading, just sample embeddings
        logger.add_embedding(torch.from_numpy(image_features)[:100], metadata=torch.from_numpy(labels)[:100], label_img=image_collection[:100])

    #feature array self indexing
    index = faiss.IndexFlatL2(feature_dim)
    index.add(image_features)

    dims, positions = index.search(image_features, K+1)

    #calculate recall@K score
    total_samples = image_features.shape[0]
    true_positive_samples = 0.0

    print ('Calculating Recall@{} Score'.format(K))
    for currentIndex in range(total_samples):
        if labels[currentIndex] in labels[positions[currentIndex]]:
            true_positive_samples += 1

    return float(true_positive_samples / float(total_samples))


def run(targetFolder, trainFile, valFile, train_transform, val_transform, model, epochs, lr, momentum, log_interval, log_dir, batchSize=256):
    train_dataset = SOPDataset(targetFolder, trainFile, transform=train_transform)
    val_dataset = SOPDataset(targetFolder, valFile, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=False)
    val_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
    
    writer = create_summary_writer(model, train_loader, log_dir)
    device = 'cpu'

    if torch.cuda.is_available():
        device = 'cuda'

    #optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
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
    
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        avg_recall = calculate_recall_score(model, train_loader)
        print("Training Results - Epoch: {}  Avg Recall: {:.2f}".format(engine.state.epoch, avg_recall))
        writer.add_scalar("training/avg_recall", avg_recall, engine.state.epoch)

        avg_recall = calculate_recall_score(model, val_loader, show_embedding=True, logger=writer)
        print("Validation Results - Epoch: {}  Avg Recall: {:.2f}".format(engine.state.epoch, avg_recall))
        writer.add_scalar("valdation/avg_recall", avg_recall, engine.state.epoch)

        if not os.path.exists('models'):
            os.mkdir('models')

        if model.module:
            torch.save(model.module, 'models/' + 'SOP_resnet18_model_epoch_' + str(engine.state.epoch) + '_recall_' + str(avg_recall) + '.pth') 
        else:
            torch.save(model, 'models/' + 'SOP_resnet18_model_epoch_' + str(engine.state.epoch) + '_recall_' + str(avg_recall) + '.pth') 

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
    parser.add_argument('--lr', type=float, default=0.002, help='learning rate (default: 0.002)')
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
