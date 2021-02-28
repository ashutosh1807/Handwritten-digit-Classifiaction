from PIL import Image
import os
import torch
from torchvision import transforms
import numpy as np
import torch.utils.data as data_utils
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import torchvision.transforms.functional as TF
from tqdm import tqdm
from colorama import Fore
import sys
from itertools import groupby
from CRNN import CRNN

digits_per_sequence = 10 
gpu = torch.device('cuda')

dataset_sequences = []
dataset_labels = [] 
transformed_random_digits_images = []

src_root = './assignment-1-dataset/dataset'
save_root = './dataset_sequence'

text = open('./assignment-1-dataset/groundTruthValues.txt')
lines = text.readlines()

for linenumber, line in enumerate(lines[1:]):
    img_name = line.split(',')[0]
    str_gt = line.split(',')[1].split('\n')[0]
    img = Image.open(os.path.join(src_root, img_name + '.png'))
    linenumber = linenumber % 16 #16 because number of lines per image to read
    bbox = (0, 32*linenumber, 320, 32*(linenumber+1))
    seq = img.crop(bbox)
    seq = transforms.Grayscale(num_output_channels=1)(seq)
    imgs_seq_lst = []
    labels_seq_lst = []
    transformed_random_digits_images = []

    for column_number,gt in enumerate(str_gt):
        bbox = (32*column_number, 0, 32*(column_number+1), 32)
        seq_c = seq.crop(bbox)
        seq_c = np.array(seq_c)

        if (gt == 'X'):
            gt = '10'
            
        labels_seq_lst.append(gt)
        imgs_seq_lst.append(seq_c)    

    imgs_seq = torch.tensor(np.array(imgs_seq_lst))
    labels_seq = torch.tensor(np.array(labels_seq_lst).astype(int))

    for img in imgs_seq:
        img = transforms.ToPILImage()(img)
        img = transforms.ToTensor()(img).numpy()
        transformed_random_digits_images.append(img)

    random_digits_images = np.array(transformed_random_digits_images)
    random_sequence = np.hstack(random_digits_images.reshape((digits_per_sequence, 32, 32)))
    random_labels = np.hstack(labels_seq.reshape(digits_per_sequence, 1))
    dataset_sequences.append(random_sequence / 255)
    dataset_labels.append(random_labels)    

dataset_data = torch.Tensor(np.array(dataset_sequences))
dataset_labels = torch.IntTensor(np.array(dataset_labels))

seq_dataset = data_utils.TensorDataset(dataset_data, dataset_labels)
train_set, val_set = torch.utils.data.random_split(seq_dataset,
                                                   [int(np.round(len(seq_dataset) * 0.8)), int(np.round(len(seq_dataset) * 0.2))])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=True)

epochs = 20
cnn_output_height = 5
cnn_output_width = 77
gru_num_layers = 2
gru_hidden_size = 64
num_classes = 12
blank_label = 11

model = CRNN(cnn_output_height, gru_hidden_size, gru_num_layers, num_classes).to(gpu)
criterion = nn.CTCLoss(blank=blank_label, reduction='mean', zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

best_valid_loss = float('inf')
for _ in range(epochs):
    train_correct = 0
    train_total = 0
    for x_train, y_train in tqdm(train_loader,
                                 position=0, leave=True,
                                 file=sys.stdout, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET)):
        batch_size = x_train.shape[0]  # x_train.shape == torch.Size([64, 28, 140])
        x_train = x_train.view(x_train.shape[0], 1, x_train.shape[1], x_train.shape[2])
        optimizer.zero_grad()
        y_pred = model(x_train.cuda())
        y_pred = y_pred.permute(1, 0, 2)  # y_pred.shape == torch.Size([64, 32, 11])
        input_lengths = torch.IntTensor(batch_size).fill_(cnn_output_width)
        target_lengths = torch.IntTensor([len(t) for t in y_train])
        loss = criterion(y_pred, y_train, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()
        _, max_index = torch.max(y_pred, dim=2)  # max_index.shape == torch.Size([32, 64])
        for i in range(batch_size):
            raw_prediction = list(max_index[:, i].detach().cpu().numpy())  # len(raw_prediction) == 32
            prediction = torch.IntTensor([c for c, _ in groupby(raw_prediction) if c != blank_label])
            if len(prediction) == len(y_train[i]) and torch.all(prediction.eq(y_train[i])):
                train_correct += 1
            train_total += 1
    print('TRAINING. Correct: ', train_correct, '/', train_total, '=', train_correct / train_total)

    val_correct = 0
    val_total = 0
    for x_val, y_val in tqdm(val_loader,
                             position=0, leave=True,
                             file=sys.stdout, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET)):
        batch_size = x_val.shape[0]
        x_val = x_val.view(x_val.shape[0], 1, x_val.shape[1], x_val.shape[2])
        y_pred = model(x_val.cuda())
        y_pred = y_pred.permute(1, 0, 2)
        input_lengths = torch.IntTensor(batch_size).fill_(cnn_output_width)
        target_lengths = torch.IntTensor([len(t) for t in y_val])
        valid_loss = criterion(y_pred, y_val, input_lengths, target_lengths)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'multidig-model.pt')
        _, max_index = torch.max(y_pred, dim=2)
        for i in range(batch_size):
            raw_prediction = list(max_index[:, i].detach().cpu().numpy())
            prediction = torch.IntTensor([c for c, _ in groupby(raw_prediction) if c != blank_label])
            if len(prediction) == len(y_val[i]) and torch.all(prediction.eq(y_val[i])):
                val_correct += 1
            val_total += 1
    print('TESTING. Correct: ', val_correct, '/', val_total, '=', val_correct / val_total)
