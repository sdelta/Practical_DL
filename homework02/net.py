import argparse
import logging
import time
import json
import os

import numpy as np

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.models
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable


logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



# https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py#L118
def create_model():
    model = nn.Sequential()
    model.add_module('conv1', nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding='same'))
    model.add_module('batchnorm_conv1', nn.BatchNorm2d(128))
    model.add_module('conv1_relu', nn.ReLU())
    model.add_module('pool1', nn.MaxPool2d(2))
    model.add_module('flatten', nn.Flatten())
    # dense "head"
    model.add_module('dense1', nn.Linear(128 * 64 * 64 // 4, 1024))
    model.add_module('batchnorm_dense1', nn.BatchNorm1d(1024))
    model.add_module('dense1_relu', nn.ReLU())
    model.add_module('dropout2', nn.Dropout(0.30))
    model.add_module('dense1_relu', nn.ReLU())
    model.add_module('dense2_logits', nn.Linear(1024, 200)) # logits for 200 classes
    return model

def compute_loss(device, model, X_batch, y_batch):
    X_batch = torch.FloatTensor(X_batch).to(device=device)
    y_batch = torch.LongTensor(y_batch).to(device=device)
    logits = model.to(device)(X_batch)
    return F.cross_entropy(logits, y_batch).mean()


def compute_accuracy(device, model, batch_gen):
    accuracy = []
    for X_batch, y_batch in batch_gen:
        logits = model(Variable(torch.FloatTensor(X_batch)).to(device))
        y_pred = logits.max(1)[1].data
        accuracy.append(np.mean( (y_batch.cpu() == y_pred.cpu()).numpy() ))
    return np.mean(accuracy)


def _train(args):
    if os.path.isdir(args.checkpoint_path):
        logger.info("Checkpointing directory {} exists".format(args.checkpoint_path))
    else:
        logger.info("Creating Checkpointing directory {}".format(args.checkpoint_path))
        os.mkdir(args.checkpoint_path)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    logger.info("Device Type: {}".format(device))

    logger.info("Loading dataset")
    


    dataset = torchvision.datasets.ImageFolder(args.train_dir, transform=transforms.ToTensor())
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [80000, 20000],
                                                           generator=torch.Generator().manual_seed(42))
    #train_dataset = torchvision.datasets.ImageFolder(args.train_dir, transform=transforms.ToTensor())
    
    test_dataset = torchvision.datasets.ImageFolder(args.test_dir, transform=transforms.ToTensor())
    train_batch_gen = torch.utils.data.DataLoader(train_dataset, 
                                                  batch_size=args.batch_size,
                                                  shuffle=True,
                                                  num_workers=args.workers)
    val_batch_gen = torch.utils.data.DataLoader(val_dataset, 
                                                  batch_size=args.batch_size,
                                                  shuffle=True,
                                                  num_workers=args.workers)
    test_batch_gen = torch.utils.data.DataLoader(test_dataset, 
                                                  batch_size=args.batch_size,
                                                  shuffle=True,
                                                  num_workers=args.workers)
    
    logger.info("Model loaded")
    model = create_model()

    if torch.cuda.device_count() > 1:
        logger.info("Gpu count: {}".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Check if checkpoints exists
    if not os.path.isfile(args.checkpoint_path + '/checkpoint.pth'):
        epoch_number = 0
    else:    
        model, optimizer, epoch_number = _load_checkpoint(model, optimizer, args)        
    
    for epoch in range(epoch_number, args.epochs):
        model.train(True)
        running_loss = 0.0
        for i, data in enumerate(train_batch_gen):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            loss = compute_loss(device, model, inputs, labels)
            #print("passed", file=sys.stderr)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:  # print every 2000 mini-batches
                logger.info('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0
        
        model.train(False)
        val_acc = compute_accuracy(device, model, val_batch_gen)
        test_acc = compute_accuracy(device, model, test_batch_gen)
        _save_checkpoint(model, optimizer, epoch + 1, loss, val_acc, test_acc, args)
            
    logger.info('Finished Training')
    return _save_model(model, args.model_dir)


def _save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, 'model.pth')
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)


def print_status(epoch, loss, val_acc, test_acc):
    logger.info("epoch {} loss {:.2f} val_acc {:.2f} % test_acc {:.2f} %".format(
        epoch, loss, val_acc * 100, test_acc * 100
    ))
    
def _save_checkpoint(model, optimizer, epoch, loss, val_acc, test_acc, args):
    print_status(epoch, loss, val_acc, test_acc)
    checkpointing_path = args.checkpoint_path + '/checkpoint.pth'
    logger.info("Saving the Checkpoint: {}".format(checkpointing_path))
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'val_acc': val_acc,
        'test_acc':test_acc
        }, checkpointing_path)

    
def _load_checkpoint(model, optimizer, args):
    logger.info("--------------------------------------------")
    logger.info("Checkpoint file found!")
    logger.info("Loading Checkpoint From: {}".format(args.checkpoint_path + '/checkpoint.pth'))
    checkpoint = torch.load(args.checkpoint_path + '/checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    logger.info("Checkpoint File Loaded")
    print_status(epoch, checkpoint['loss'], checkpoint['val_acc'], checkpoint['test_acc'])
    logger.info('Resuming training from epoch: {}'.format(epoch))
    logger.info("--------------------------------------------")
    return model, optimizer, epoch

    
def model_fn(model_dir):
    logger.info('model_fn')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Net()
    if torch.cuda.device_count() > 1:
        logger.info("Gpu count: {}".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--workers', type=int, default=2, metavar='W',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', type=int, default=20, metavar='E',
                        help='number of total epochs to run (default: 2)')
    parser.add_argument('--batch_size', type=int, default=128, metavar='BS',
                        help='batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                        help='initial learning rate (default: 0.00001)')

    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test-dir', type=str, default=os.environ['SM_CHANNEL_VAL'])
    parser.add_argument("--checkpoint-path",type=str,default="/opt/ml/checkpoints")
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    _train(parser.parse_args())
