import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

import pandas as pd
from sklearn.model_selection import train_test_split

from utils import *
from prepare_data import prepare_testing

seed = 0
torch.manual_seed(seed)


def train(prefix, device):
    df = pd.read_csv(prefix+'data/train_processed.csv')
    train_data, valid_data = train_test_split(df, test_size=0.15, random_state=seed)

    transform = transforms.Compose([transforms.ToTensor()])
    trainset = DNA(train_data, transform=transform)
    validset = DNA(valid_data, transform=transform)

    batch_size = 64
    print_freq = 200
    num_epochs = 20
    learning_rate = 1e-3
    weight_decay = 1e-3

    trainloader = torch.utils.data.DataLoader(trainset, 
        batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    validloader = torch.utils.data.DataLoader(validset, 
        batch_size=len(validset), shuffle=False, num_workers=2, pin_memory=True)

    net = Net().float().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    outputs = labels = None
    train_loss = []
    valid_loss = []
    valid_steps = []

    path = prefix+'results_training'
    os.makedirs(path, exist_ok=True)

    vs = 0
    best_valid_loss = 99
    best_outputs = None
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            fw_inputs, rev_inputs, labels = data
            fw_inputs, rev_inputs, labels = fw_inputs.to(device), rev_inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(fw_inputs.float(), rev_inputs.float())
            loss = criterion(outputs, labels.view(-1, 1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % print_freq == print_freq-1:    # print every 200 mini-batches
                print('[%d, %5d] training loss: %.3f' % (epoch + 1, i + 1, running_loss / print_freq))
                running_loss = 0.0

            train_loss.append(loss.item())
            vs += 1

        # validation every epoch
        with torch.no_grad():
            for data in validloader:
                fw_inputs, rev_inputs, labels = data
                fw_inputs, rev_inputs, labels = fw_inputs.to(device), rev_inputs.to(device), labels.to(device)
                outputs = net(fw_inputs.float(), rev_inputs.float())
                loss = criterion(outputs, labels.view(-1, 1))
                print('[%d] validation loss: %.3f' % (epoch + 1, loss.item()))

                valid_loss.append(loss.item())
                valid_steps.append(vs)

                if valid_loss[epoch] < best_valid_loss:    
                    best_valid_loss = valid_loss[epoch]
                    best_outputs = outputs.detach().clone()
                    print('Best validation loss so far, saving model')
                    torch.save(net.state_dict(), path+'/tf.pth')

    print('Finished Training')

    labels = labels.cpu()
    predicted = torch.tensor([1. if d > 0.5 else 0. for d in best_outputs.data])
    print('Best validation loss: %.3f' % (best_valid_loss))
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    print('Accuracy on the validation sequences: %d %%' % (100 * correct / total))

    # generate ROC, PR, Training Curves
    roc_pr(labels, predicted, path)
    training_curve(train_loss, valid_loss, valid_steps, path, smoothing_weight=0.9)

    # generate sequence motifs
    net = Net().float().to(device)
    net.load_state_dict(torch.load(path+'/tf.pth'))
    generate_motfis(net, validloader, path, device)


def test(prefix, path, device):
    df = prepare_testing(path)

    transform = transforms.Compose([transforms.ToTensor()])
    testset = DNA(df, transform=transform)

    testloader = torch.utils.data.DataLoader(testset, 
        batch_size=len(testset), shuffle=False, num_workers=2, pin_memory=True)

    net = Net().float().to(device)
    net.load_state_dict(torch.load('results_training/tf.pth'))    
    criterion = nn.BCELoss()

    outputs = labels = None

    with torch.no_grad():
        for data in testloader:
            fw_inputs, rev_inputs, labels = data
            outputs = net(fw_inputs.float(), rev_inputs.float())
            loss = criterion(outputs, labels.view(-1, 1))
            print('Test loss: %.3f' % (loss.item()))

    print('Finished Testing')

    predicted = torch.tensor([1. if d > 0.5 else 0. for d in outputs.data])
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    print('Accuracy on the test sequences: %d %%' % (100 * correct / total))

    # generate ROC, PR Curves and sequence motifs
    path = prefix+'results_test'
    os.makedirs(path, exist_ok=True)
    roc_pr(labels, predicted, path)
    generate_motfis(net, testloader, path, device)


def main(args):
    prefix = ''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.test:
        path = prefix+'data/test.csv'
        test(prefix, path, device)
    else:
        train(prefix, device)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='TF')
    parser.add_argument('--test', default=False, action='store_true', help='test the model')
    args = parser.parse_args()

    main(args)


