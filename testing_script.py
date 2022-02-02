import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
import glob
from glob import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import datasets, models, transforms
import time
import copy
from PIL import Image
import datetime
# for progress bar
from tqdm import tqdm_notebook, tqdm, trange
#to print multiple outputs
# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity='all'

# from pytorch_pretrained_vit import ViT

# from linformer import Linformer
from torch.utils.tensorboard import SummaryWriter


class CustomDataset(Dataset):
    def __init__(self, labels_df, img_path, transform=None):
        self.labels_df = labels_df
        self.img_path = img_path
        self.transform = transform

    def __len__(self):
        return self.labels_df.shape[0]

    def __getitem__(self, idx):
        image_name = self.labels_df.filename[idx]
        for file in self.img_path:
            if file.endswith(image_name + '.png'):
                img = Image.open(file)
        #                 print(file)
        label = self.labels_df.label_idx[idx]
        #         print(file)

        if self.transform:
            img = self.transform(img)

        return img, label


PATH = "/home/tzortzis/real_data/Downloads/"
EXT = "*.csv"
img_EXT = "*.png"

test_path = "/home/tzortzis/real_data/Downloads/rectangular_stripes/"


all_csv_files = []
all_img_files = []
for path, subdir, files in os.walk(PATH):
    for file in glob(os.path.join(path, EXT)):
        if file.endswith('data.csv'):
            all_csv_files.append(file)
    for i in glob(os.path.join(path, img_EXT)):
        all_img_files.append(i)


li = []

for filename in all_csv_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)



frame = pd.concat(li, axis=0, ignore_index=True)


state = frame.cloth_state.unique()
state2idx = dict((state,idx) for idx, state in enumerate(state))
idx2state = dict((idx,state) for idx,state in enumerate(state))
frame['label_idx'] = [state2idx[b] for b in frame.cloth_state]

training_dataset = CustomDataset(frame, all_img_files)

input_size = 224
bs = 64

data_transforms = {
    'train': transforms.Compose([
    transforms.Resize(input_size),
    transforms.RandomResizedCrop(input_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])#these vectors is the mean and the std from the statistics in imagenet. They are always the same as far as I can recall
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_img = []
test_filename = []
for string in all_img_files:
    if test_path in string:
        test_img.append(string)
for f in frame.filename:
    for string in test_img:
        if f in string:
            test_filename.append(f)

test_frame = frame.loc[frame['filename'].isin(test_filename)].reset_index()

val_dataset = CustomDataset(test_frame, test_img, transform=data_transforms['val'])
image_dataset = {'val':val_dataset}
dataset_names = ['val']
image_dataloader = {x:DataLoader(image_dataset[x],batch_size=bs,shuffle=True,num_workers=6) for x in dataset_names}
dataset_sizes = {x:len(image_dataset[x]) for x in dataset_names}


from efficientnet_pytorch import EfficientNet
model_ft = EfficientNet.from_pretrained('efficientnet-b0')

for param in model_ft.parameters():
    param.requires_grad = True    # By doing this I am keeping the parameters of the feature layers frozen so they won't update

num_fc_ftr = model_ft._fc.in_features
model_ft._fc = nn.Sequential(
    nn.Linear(num_fc_ftr, 1000),
    nn.BatchNorm1d(1000),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(1000,500),
    nn.BatchNorm1d(500),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(500,10))


model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# optimizer = optim.Adam([
#     {'params':model_ft.fc.parameters()}
# ], lr=0.001)
optimizer = optim.SGD(model_ft.parameters(), lr=0.1)

scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.01, last_epoch=-1)


model_ft.load_state_dict(torch.load('/home/tzortzis/all_real_eff-finetuned-no-stripes.pt'))

model_ft.eval()  # it-disables-dropout
from torch.nn import functional as F
import torchvision.transforms.functional as TF



def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    basename = "misclass"
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    filename = "_".join([basename, suffix])

    plt.savefig(filename)

with torch.no_grad():
    correct = 0
    total = 0
    phase = 'val'
    predictions = []
    preds = []
    probabilities = []
    indx = []
    top = []
    wrong_predictions = []
    wrong_images = []
    true_labels = []
    top2_pred_labels = []

    images_so_far = 0
    fig = plt.figure()

    num_images = 6

    nb_classes = 10
    title = None

    confusion_matrix = torch.zeros(nb_classes, nb_classes)

    for data in image_dataloader['val']:
        #         images = images.to(device)
        #         labels = labels.to(device)
        #         outputs = model(images)

        inps, labels = data
        inps = inps.to(device)
        labels = labels.to(device)
        outputs = model_ft(inps)
        sm = torch.nn.Softmax()

        # import pdb
        # pdb.set_trace()

        _, predicted = torch.max(outputs.data, 1)
        top2, pred_labels = torch.topk(sm(outputs.data), 2, dim=1, largest=True, sorted=True)
        top5, pred_labels5 = torch.topk(sm(outputs.data), 5, dim=1, largest=True, sorted=True)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # import pdb
        # pdb.set_trace()

        # for j in range(inps.size()[0]):
        #     images_so_far += 1
        #     ax = plt.subplot(num_images // 2, 2, images_so_far)
        #     ax.axis('off')
        #     ax.set_title('predicted: {}'.format(labels[predicted[j]]))
        #     imshow(inps.cpu().data[j])

        wrong_idx = (predicted != labels.view_as(predicted)).nonzero()[:, 0]
        wrong_samples = inps[wrong_idx]
        wrong_preds = predicted[wrong_idx]
        actual_preds = labels.view_as(predicted)[wrong_idx]

        for j in range(inps[wrong_idx].size()[0]):
        #     images_so_far += 1
        #     ax = plt.subplot(num_images // 2, 2, images_so_far)
        #     ax.axis('off')
            # ax.set_title('predicted: {}'.format(labels[predicted[wrong_idx][j]]))
            imshow(inps[wrong_idx].cpu().data[j])
        #     if title is not None:
        #         plt.title(title)
        #     plt.pause(0.001)  # pause a bit so that plots are updated

        indx.append(actual_preds)
        top.append(top2[wrong_idx])
        top2_pred_labels.append(pred_labels[wrong_idx])
        wrong_images.append(wrong_samples)
        wrong_predictions.append(wrong_preds)
        true_labels.append(actual_preds)

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])



        #         print(val_df.iloc[wrong_idx])
        #         for i in range(len(wrong_idx)):
        #             sample = wrong_samples[i]
        #             wrong_pred = wrong_preds[i]
        #             actual_pred = actual_preds[i]
        #             # Undo normalization
        # #             sample = sample * 0.3081
        # #             sample = sample + 0.1307
        # #             sample = sample * 255.
        #             sample = sample.byte()
        #             img = TF.to_pil_image(sample)
        #             img.save('/home/tzortzis/preds/wrong_idx{}_pred{}_actual{}.png'.format(
        #                 wrong_idx[i], wrong_pred.item(), actual_pred.item()))
        #             print('wrong_idx{}_pred{}_actual{}'.format(wrong_idx[i], wrong_pred.item(), actual_pred.item()))
        # #             print(frame.iloc(wrong_idx[i]))

        for t, p in zip(labels.view(-1), predicted.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

    print('Test Accuracy of the model: {} %'.format(100 * correct / total))
#     print('wrong_idx{}_pred{}_actual{}'.format(wrong_idx[i], wrong_pred.item(), actual_pred.item()))

