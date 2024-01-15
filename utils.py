import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import logomaker
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)


class DNA(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = np.array(list(self.data.iloc[idx, 0]))

        seq[seq=='A']=0
        seq[seq=='T']=1
        seq[seq=='C']=2
        seq[seq=='G']=3
        seq[seq=='N']=4

        seq = seq.astype(int)

        fwd_img = np.zeros((5, seq.size))
        fwd_img[seq, np.arange(seq.size)] = 1
        fwd_img = fwd_img[0:4,:]

        # reverse complement 
        rev_seq = np.array(list(self.data.iloc[idx, 0]))
        rev_seq = np.flip(rev_seq)

        rev_seq[rev_seq=='A']=1
        rev_seq[rev_seq=='T']=0
        rev_seq[rev_seq=='C']=3
        rev_seq[rev_seq=='G']=2
        rev_seq[rev_seq=='N']=4

        rev_seq = rev_seq.astype(int)

        rev_cmp_img = np.zeros((5, rev_seq.size))
        rev_cmp_img[rev_seq, np.arange(rev_seq.size)] = 1
        rev_cmp_img = rev_cmp_img[0:4, :]

        label = torch.tensor(float(self.data.iloc[idx, 1]))

        if self.transform:
            fwd_img = self.transform(fwd_img)
            rev_cmp_img = self.transform(rev_cmp_img)

        return fwd_img, rev_cmp_img, label


def weights_init(m):
    # intializes weights based on the Xavier initialization
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.fill_(0)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, (4, 24))
        self.batchnorm = nn.BatchNorm1d(64)
        self.fc1 = nn.Linear(64, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)

        self.apply(weights_init)

    def forward(self, x1, x2):
        x1 = F.relu(self.conv(x1))
        x2 = F.relu(self.conv(x2))
        kernel_size = (1, x1.shape[-1])
        x1 = torch.cat([torch.flatten(F.max_pool2d(x1, kernel_size), 1),  
                        torch.flatten(F.avg_pool2d(x1, kernel_size), 1)], 1)
        x2 = torch.cat([torch.flatten(F.max_pool2d(x2, kernel_size), 1), 
                        torch.flatten(F.avg_pool2d(x2, kernel_size), 1)], 1)
        x = torch.cat([x1, x2], 1)
        x = self.batchnorm(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

    def motif(self, x1, x2):
        # function to generate motif detector's response based on the input
        x1 = F.relu(self.conv(x1))
        x2 = F.relu(self.conv(x2))
        return x1, x2


def smooth(scalars, weight):
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def generate_motfis(net, testloader, path, device):
    with torch.no_grad():
        for data in testloader:
            fwd_inputs, rev_inputs, labels = data
            fwd_inputs, rev_inputs, labels = fwd_inputs.to(device), rev_inputs.to(device), labels.to(device)
            x1, x2 = net.motif(fwd_inputs.float(), rev_inputs.float())

    fwd_inputs = fwd_inputs.cpu().numpy()
    x1 = x1.data.cpu().numpy()
    x2 = x2.data.cpu().numpy()

    num_seqs, num_filters, _, _ = x1.shape
    filter_len = 24

    # extracting subsequences based on argmax position
    subseqs = []
    for i in range(num_seqs):
        if np.max(x1[i]) > np.max(x2[i]):
            start_pos = np.unravel_index(x1[i].argmax(), x1[i].shape)[2]
        else:
            start_pos = np.unravel_index(x2[i].argmax(), x2[i].shape)[2]
        end_pos = start_pos + filter_len
        subseqs.append(fwd_inputs[i, 0, :, start_pos:end_pos])

    subseqs = np.asarray(subseqs)
    pos_freq = np.sum(subseqs, axis=0)

    # calculating the PWM from the PFM, also calculating the Information Content
    ic = []
    for i in range(filter_len):
        pos_sum = np.sum(pos_freq[:, i])
        pos_freq[:, i] /= pos_sum 
        entropy = -sum([pos_freq[j, i]*np.log2(pos_freq[j, i]) for j in range(4)])
        small_sample_corr = 3./(np.log(2)*2*num_seqs)
        ic.append(2. - (entropy+small_sample_corr))
        pos_freq[:, i] *= ic[i]

    # plotting the sequence logo
    pos_freq = pos_freq.T
    pos_freq_df = pd.DataFrame(pos_freq, columns = ['A', 'T', 'C', 'G'])

    plt.figure()
    nn_logo = logomaker.Logo(pos_freq_df)
    nn_logo.style_spines(visible=False)
    nn_logo.ax.set_xlim([-1, 25])
    nn_logo.ax.set_xticks([])
    nn_logo.ax.set_yticks([])
    nn_logo.ax.set_ylim([0, max(ic)])

    plt.savefig(path+'/sequence_logo.png')
    plt.close()
    # plt.show()


def roc_pr(labels, predictions, path):
    plt.figure()
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver-Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(path+'/roc_curve.png')
    plt.close()
    # plt.show()

    plt.figure()
    precision, recall, thresholds = precision_recall_curve(labels, predictions)
    plt.plot(recall, precision, label='PR curve (area = %0.3f)' % auc(recall, precision))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall Curve')
    plt.legend(loc="lower right")
    plt.savefig(path+'/pr_curve.png')
    plt.close()
    # plt.show()


def training_curve(train_loss, valid_loss, valid_steps, path, smoothing_weight):
    plt.figure()
    train_loss = smooth(train_loss, weight=smoothing_weight)
    plt.plot(range(valid_steps[-1]), train_loss, label='Training Cruve')
    plt.plot(valid_steps, valid_loss, label='Validation Cruve')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training / Validation Curve')
    plt.legend(loc="upper right")
    plt.savefig(path+'/training_curve.png')
    plt.close()
    # plt.show()
