import numpy as np
import operator
from pathlib import Path
from sklearn.utils import shuffle
from statistics import mean, stdev
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import gc
from sklearn.metrics.pairwise import cosine_similarity

import utils_processing

class LR(nn.Module):

    def __init__(self, num_classes):
        super(LR, self).__init__()
        self.fc1 = nn.Linear(768, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        output = torch.sigmoid(x)
        # output = torch.softmax(x, dim=1)
        return output

class MLP(nn.Module):

    def __init__(self, num_classes, num_classes_aux):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(768, 50)
        self.relu1 = nn.Tanh()
        self.fc2 = nn.Linear(50, num_classes + num_classes_aux)
        self.num_classes = num_classes
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        # output = torch.sigmoid(x)
        # output = torch.softmax(x, dim=1)
        x_num_classes = torch.softmax(x[:, :self.num_classes], dim = 1)
        x_aux = torch.softmax(x[:, self.num_classes:], dim = 1)
        output = torch.cat([x_num_classes, x_aux], dim = 1)
        return output

def train_mlp(  
        setup,
        train_txt_path,
        train_embedding_path,
        test_txt_path,
        test_embedding_path,
        num_classes,
        train_size,
        seed_num,
        alpha,
        minibatch_size,
        num_epochs,
        ):

    torch.manual_seed(seed_num)
    np.random.seed(seed_num)
    
    # get all the data
    train_x, train_y, train_y_aux, num_classes_aux = utils_processing.get_split_train_x_y(train_txt_path, train_size, seed_num, setup, alpha)
    test_x, test_y = utils_processing.get_x_y(test_txt_path, test_embedding_path)
    # print(train_x.shape, train_y.shape, train_y_aux.shape, test_x.shape, test_y.shape)

    model = MLP(num_classes=num_classes, num_classes_aux=num_classes_aux)
    optimizer = optim.Adam(params=model.parameters(), lr=0.001, weight_decay=0.05) #wow, works for even large learning rates
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)

    num_minibatches_train = int(train_x.shape[0] / minibatch_size)
    val_acc_list = []

    ######## training loop ########
    for epoch in range(1, num_epochs + 1):

        ######## training ########
        model.train(mode=True)

        train_x, train_y, train_y_aux = shuffle(train_x, train_y, train_y_aux, random_state = seed_num)

        for minibatch_num in range(num_minibatches_train):

            start_idx = minibatch_num * minibatch_size
            end_idx = start_idx + minibatch_size
            train_inputs = torch.from_numpy(train_x[start_idx:end_idx].astype(np.float32))
            train_labels = torch.from_numpy(train_y[start_idx:end_idx].astype(np.long))
            train_labels_aux = torch.from_numpy(train_y_aux[start_idx:end_idx].astype(np.long))

            # Forward and backpropagation.
            with torch.set_grad_enabled(mode=True):

                combined_outputs = model(train_inputs)
                primary_outputs = combined_outputs[:, :num_classes]
                train_conf, train_preds = torch.max(primary_outputs, dim=1)

                aux_outputs = combined_outputs[:, num_classes:]

                train_loss = nn.CrossEntropyLoss()(input=primary_outputs, target=train_labels)
                aux_loss = nn.CrossEntropyLoss()(input=aux_outputs, target=train_labels_aux)
                _, aux_preds = torch.max(aux_outputs, dim=1)
                aux_acc = accuracy_score(train_y_aux[start_idx:end_idx], aux_preds)

                combined_loss = train_loss + aux_loss if 'mtl' in setup else train_loss
                combined_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        ######## validation ########
        model.train(mode=False)

        val_inputs = torch.from_numpy(test_x.astype(np.float32))
        val_labels = torch.from_numpy(test_y.astype(np.long))

        # Feed forward.
        with torch.set_grad_enabled(mode=False):
            val_outputs = model(val_inputs)[:, :num_classes]
            val_confs, val_preds = torch.max(val_outputs, dim=1)
            val_loss = nn.CrossEntropyLoss()(input=val_outputs, target=val_labels)
            val_loss_print = val_loss / val_inputs.shape[0]
            val_acc = accuracy_score(test_y, val_preds)
            val_acc_list.append(val_acc)

    gc.collect()
    return mean(val_acc_list[-5:])

def train_mlp_multiple(  
    setup,
    train_txt_path,
    train_embedding_path,
    test_txt_path,
    test_embedding_path,
    num_classes,
    dataset_name,
    train_size,
    num_seeds,
    alpha,
    minibatch_size,
    num_epochs = 100,
    ):

    val_acc_list = []

    for seed_num in range(num_seeds):

        val_acc = train_mlp(  
            setup,
            train_txt_path,
            train_embedding_path,
            test_txt_path,
            test_embedding_path,
            num_classes,
            train_size,
            seed_num,
            alpha,
            minibatch_size,
            num_epochs,
            )

        val_acc_list.append(val_acc)

    val_acc_stdev = stdev(val_acc_list) if len(val_acc_list) >= 2 else -1 
    return mean(val_acc_list), val_acc_stdev