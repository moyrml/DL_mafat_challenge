#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Optional: mounting to Google Drive to read the data files.
# from google.colab import drive
mount_path = '.'
# drive.mount(mount_path)
notebook_root = '.'

# In[6]:


import os
import random
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

from plotly.subplots import make_subplots
import plotly.graph_objects as go

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import DataLoader




import torch
import torch.nn as nn

import numpy as np
import matplotlib.gridspec as gridspec

# get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

competition_path = 'Data'


# Functions for loading the data
def load_data(file_path):
    """
    Reads all data files (metadata and signal matrix data) as python dictionary,
    the pkl and csv files must have the same file name.

    Arguments:
      file_path -- {str} -- path to the iq_matrix file and metadata file

    Returns:
      Python dictionary
    """
    pkl = load_pkl_data(file_path)
    meta = load_csv_metadata(file_path)
    data_dictionary = {**meta, **pkl}

    for key in data_dictionary.keys():
        data_dictionary[key] = np.array(data_dictionary[key])

    return data_dictionary


def load_pkl_data(file_path):
    """
    Reads pickle file as a python dictionary (only Signal data).

    Arguments:
      file_path -- {str} -- path to pickle iq_matrix file

    Returns:
      Python dictionary
    """
    path = os.path.join(mount_path, competition_path, file_path + '.pkl')
    with open(path, 'rb') as data:
        output = pickle.load(data)
    return output


def load_csv_metadata(file_path):
    """
    Reads csv as pandas DataFrame (only Metadata).

    Arguments:
      file_path -- {str} -- path to csv metadata file

    Returns:
      Pandas DataFarme
    """
    path = os.path.join(mount_path, competition_path, file_path + '.csv')
    with open(path, 'rb') as data:
        output = pd.read_csv(data)
    return output

    # Function for splitting the data to training and validation


# and function for selecting samples of segments from the Auxiliary dataset
def split_train_val(data):
    """
    Split the data to train and validation set.
    The validation set is built from training set segments of
    geolocation_id 1 and 4.
    Use the function only after the training set is complete and preprocessed.

    Arguments:
      data -- {ndarray} -- the data set to split

    Returns:
      iq_sweep_burst ndarray matrices
      target_type vector
      for training and validation sets
    """
    idx = ((data['geolocation_id'] == 4) | (data['geolocation_id'] == 1)) & (data['segment_id'] % 6 == 0)
    training_x = data['iq_sweep_burst'][np.logical_not(idx)]
    training_y = data['target_type'][np.logical_not(idx)]
    validation_x = data['iq_sweep_burst'][idx]
    validation_y = data['target_type'][idx]
    return training_x, training_y, validation_x, validation_y


def stratified_split_train_val(data):
    indices = np.arange(len(data['target_type']))

    from sklearn.model_selection import train_test_split
    train_inds, val_inds = train_test_split(
        indices,
        stratify=train_df['target_type'],
        train_size=0.8
    )

    training_x = data['iq_sweep_burst'][train_inds]
    training_y = data['target_type'][train_inds]
    validation_x = data['iq_sweep_burst'][val_inds]
    validation_y = data['target_type'][val_inds]
    return training_x, training_y, validation_x, validation_y


def stratified_K_split_train_val(data):
    indices = np.arange(len(data['target_type']))

    from sklearn.model_selection import train_test_split
    train_inds, val_inds = train_test_split(
        indices,
        stratify=train_df['target_type'],
        train_size=0.8
    )

    training_x = data['iq_sweep_burst'][train_inds]
    training_y = data['target_type'][train_inds]
    validation_x = data['iq_sweep_burst'][val_inds]
    validation_y = data['target_type'][val_inds]
    return training_x, training_y, validation_x, validation_y


def aux_split(data):
    """
    Selects segments from the auxilary set for training set.
    Takes the first 3 segments (or less) from each track.

    Arguments:
      data {dataframe} -- the auxilary data

    Returns:
      The auxilary data for the training
    """
    idx = np.bool_(np.zeros(len(data['track_id'])))
    for track in np.unique(data['track_id']):
        idx |= data['segment_id'] == (data['segment_id'][data['track_id'] == track][:3])

    for key in data:
        data[key] = data[key][idx]
    return data

    # The function append_dict is for concatenating the training set


# with the Auxiliary data set segments

def append_dict(dict1, dict2):
    for key in dict1:
        dict1[key] = np.concatenate([dict1[key], dict2[key]], axis=0)
    return dict1

    # Functions for preprocessing and preprocess function


def fft(iq, axis=0):
    """
    Computes the log of discrete Fourier Transform (DFT).

    Arguments:
      iq_burst -- {ndarray} -- 'iq_sweep_burst' array
      axis -- {int} -- axis to perform fft in (Default = 0)

    Returns:
      log of DFT on iq_burst array
    """
    iq = np.log(np.abs(np.fft.fft(hann(iq), axis=axis)))
    return iq


def hann(iq, window=None):
    """
    Preformes Hann smoothing of 'iq_sweep_burst'.

    Arguments:
      iq {ndarray} -- 'iq_sweep_burst' array
      window -{range} -- range of hann window indices (Default=None)
               if None the whole column is taken

    Returns:
      Regulazied iq in shape - (window[1] - window[0] - 2, iq.shape[1])
    """
    if window is None:
        window = [0, len(iq)]

    N = window[1] - window[0] - 1
    n = np.arange(window[0], window[1])
    n = n.reshape(len(n), 1)
    hannCol = 0.5 * (1 - np.cos(2 * np.pi * (n / N)))
    return (hannCol * iq[window[0]:window[1]])[1:-1]


def max_value_on_doppler(iq, doppler_burst):
    """
    Set max value on I/Q matrix using doppler burst vector.

    Arguments:
      iq_burst -- {ndarray} -- 'iq_sweep_burst' array
      doppler_burst -- {ndarray} -- 'doppler_burst' array (center of mass)

    Returns:
      I/Q matrix with the max value instead of the original values
      The doppler burst marks the matrix values to change by max value
    """
    iq_max_value = np.max(iq)
    for i in range(iq.shape[1]):
        if doppler_burst[i] >= len(iq):
            continue
        iq[doppler_burst[i], i] = iq_max_value
    return iq


def normalize(iq):
    """
    Calculates normalized values for iq_sweep_burst matrix:
    (vlaue-mean)/std.
    """
    m = iq.mean()
    s = iq.std()
    return (iq - m) / s


def data_preprocess(data):
    """
    Preforms data preprocessing.
    Change target_type lables from string to integer:
    'human'  --> 1
    'animal' --> 0

    Arguments:
      data -- {ndarray} -- the data set

    Returns:
      processed data (max values by doppler burst, DFT, normalization)
    """
    X = []
    for i in range(len(data['iq_sweep_burst'])):
        iq = fft(data['iq_sweep_burst'][i])
        iq = max_value_on_doppler(iq, data['doppler_burst'][i])
        iq = normalize(iq)
        X.append(iq)

    data['iq_sweep_burst'] = np.array(X)
    if 'target_type' in data:
        data['target_type'][data['target_type'] == 'animal'] = 0
        data['target_type'][data['target_type'] == 'human'] = 1
    return data



num_classes = 2
embedding = nn.Embedding(
    num_classes,  # num classes
    96,  # embedding dim
)


def vec2idx(vec, embedding):
    dists = torch.mm(vec, embedding.weight.data.T)
    return dists


class model_D(nn.Module):
    def __init__(self, embedding=None, batch_size=128):
        super().__init__()

        self.embedding = embedding

        self.D = nn.Sequential(Unflatten(batch_size, 1, 126, 32),
                               nn.Conv2d(1, 32, kernel_size=5, stride=1),
                               nn.LeakyReLU(inplace=False),
                               nn.MaxPool2d(2, 2),
                               nn.Conv2d(32, 64, kernel_size=5, stride=1),
                               nn.LeakyReLU(inplace=False),
                               nn.MaxPool2d(2, 2),
                               Flatten(),
                               nn.Linear(8960, 1024),
                               nn.LeakyReLU(inplace=False),
                               # nn.Linear(1024,1)
                               )

        self.output_real_fake = nn.Linear(1024, 1)
        self.output_class = nn.Linear(1024, 96)

    def forward(self, x):
        x = self.D(x)

        real_fake = self.output_real_fake(x)
        c = self.output_class(x)

        if self.embedding is not None:
            c = vec2idx(c, self.embedding)

            return real_fake, c
        return real_fake


class model_G(nn.Module):
    def __init__(self, embedding=None, batch_size=128):
        super().__init__()

        self.embedding = embedding

        self.G = nn.Sequential(
            nn.Linear(96 * 2, 2048),
            nn.ReLU(inplace=False),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 6272),
            nn.ReLU(inplace=False),
            nn.BatchNorm1d(6272),
            Unflatten(batch_size, 128, 7, 7),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, kernel_size=(3, 5), stride=(2, 1), padding=(1, 0)),
            nn.ReLU(inplace=False),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, kernel_size=(5, 1), stride=(2, 1), padding=(1, 0), dilation=(5, 1)),

        )

        self.flatten = Flatten()

    def forward(self, x, y=None, flatten=True):
        if y is None:
            y = torch.zeros_like(x).to(device)
        if self.embedding is not None:
            y = self.embedding(y.long())

        x = torch.cat([x, x], dim=1)
        # x = torch.cat([x,y.squeeze()],dim=1)
        x = self.G(x)

        if not flatten:
            return x[:, :, :126, :]

        return self.flatten(x[:, :, :126, :])


from torch.utils.data import DataLoader, Dataset


class IQ_data(Dataset):
    def __init__(self, data_x, data_y, transforms=None, balance=False):
        super().__init__()

        self.data_x = data_x
        self.data_y = data_y
        self.transforms = transforms

        if balance:
            pos = (data_y == 1).sum()
            neg = np.where(data_y == 0)[0]
            neg_to_take = np.random.permutation(neg)[:pos]

            to_take = list(neg_to_take) + list(np.where(data_y == 1)[0])
            self.data_x = self.data_x[to_take]
            self.data_y = self.data_y[to_take]

    def __len__(self):
        return self.data_x.shape[0]

    def __getitem__(self, idx):
        datapoint = self.data_x[idx]

        if self.transforms is not None:
            datapoint = self.transforms(datapoint)

        datapoint = torch.Tensor(datapoint)
        datapoint = datapoint.permute(2, 0, 1)

        if self.data_y is not None:
            label = self.data_y[idx]
            return datapoint, label

        return datapoint

class unflatten(nn.Module):
    '''
    This is required if running torch version <1.7
    '''

    def __init__(self, target_dim=(256, 4, 4)):
        super().__init__()
        self.target_dim = target_dim

    def forward(self, x):
        '''
        x: torch tensor of shape [BS, C, L]
        '''
        return x.view([x.shape[0], *self.target_dim])


class Unflatten(nn.Module):
    """
    An Unflatten module receives an input of shape (N, C*H*W) and reshapes it
    to produce an output of shape (N, C, H, W).
    """

    def __init__(self, N=-1, C=128, H=7, W=7):
        super(Unflatten, self).__init__()
        self.N = N
        self.C = C
        self.H = H
        self.W = W

    def forward(self, x):
        return x.view(self.N, self.C, self.H, self.W)


class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size()  # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image


def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function.

    As per https://github.com/pytorch/pytorch/issues/751
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

    Inputs:
    - input: PyTorch Tensor of shape (N, ) giving scores.
    - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.

    Returns:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of input data.
    """
    neg_abs = - input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()


def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss described above.

    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
    N, _ = logits_real.size()
    loss = (bce_loss(logits_real, torch.ones(N).to(device))) + (bce_loss(logits_fake, torch.zeros(N).to(device)))
    return loss


def generator_loss(logits_fake):
    """
    Computes the generator loss described above.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    N, _ = logits_fake.size()
    loss = (bce_loss(logits_fake, torch.ones(N).to(device)))
    return loss


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def generate_samples(G, batch_size, num_samples=[512], file_prefix='fake_images'):
    num_iteration = np.ceil(max(num_samples) / float(batch_size)).astype(int)

    g_fake_class = torch.randint(0, num_classes, (batch_size, 1)).to(device)
    g_fake_seed = torch.randn(batch_size, 96).to(device)
    fake_images = G(g_fake_seed, g_fake_class, flatten=False)

    for i in range(num_iteration-1):
        g_fake_class = torch.randint(0, num_classes, (batch_size, 1)).to(device)
        g_fake_seed = torch.randn(batch_size, 96).to(device)
        fake_images = torch.cat([fake_images,G(g_fake_seed, g_fake_class, flatten=False)], dim=0)


    for n in num_samples:
        to_take = np.random.choice(
            np.arange(fake_images.shape[0]),
            n
        )
        with open(f'{file_prefix}_{n}.pkl', 'wb') as f:
            pickle.dump(fake_images[to_take, ...], f)


def show_images(images):
    images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg, sqrtimg]))
    return


if __name__ == '__main__':
    # Training set
    train_path = 'MAFAT RADAR Challenge - Training Set V1'
    training_df = load_data(train_path)

    # Adding segments from the experiment auxiliary set to the training set
    train_df = training_df
    # train_df = append_dict(training_df, train_aux)

    # Preprocessing and split the data to training and validation
    train_df = data_preprocess(train_df.copy())
    # train_x, train_y, val_x, val_y = stratified_split_train_val(train_df)
    train_x = train_df['iq_sweep_burst']
    train_y = train_df['target_type']

    # val_y =  val_y.astype(int)
    # train_y =train_y.astype(int)
    # train_x = train_x.reshape(list(train_x.shape)+[1])
    # val_x = val_x.reshape(list(val_x.shape)+[1])

    trainval_y = train_y.astype(int)
    trainval_x = train_x.reshape(list(train_x.shape) + [1])
    del train_x, train_y, training_df, train_df

    # trainval_pos_y_idx = np.where(trainval_y == 1)[0]
    # pos_y = trainval_y[trainval_pos_y_idx]
    # pos_x = trainval_x[trainval_pos_y_idx, ...]

    trainval_neg_y_idx = np.where(trainval_y == 0)[0]
    neg_y = trainval_y[trainval_neg_y_idx]
    neg_x = trainval_x[trainval_neg_y_idx, ...]
    del trainval_x, trainval_y

    # train_set = IQ_ata(train_x, train_y, balance = True)
    # valid_set = IQ_data(val_x, val_y)
    # trainval_set = IQ_data(trainval_x, trainval_y)
    # trainval_pos_set = IQ_data(pos_x, pos_y)
    trainval_neg_set = IQ_data(neg_x, neg_y)
    # In[9]:

    init_dim = np.array([64, 8, 8])
    noise_vect = torch.randn(2, 1, np.prod(init_dim))




    # In[ ]:


    num_epochs = 1000
    batch_size = 128
    show_every = 100


    # trainval_loader = DataLoader(trainval_set, batch_size=batch_size, shuffle=True, num_workers=2)
    # trainval_loader = DataLoader(trainval_pos_set, batch_size=batch_size, shuffle=True, num_workers=2)
    trainval_loader = DataLoader(trainval_neg_set, batch_size=batch_size, shuffle=True, num_workers=2)

    D = model_D(embedding, batch_size=batch_size).to(device)
    G = model_G(embedding, batch_size=batch_size).to(device)

    # raise ValueError()

    G.apply(weights_init)
    D.apply(weights_init)

    lr = 1e-4
    beta1 = 0.5
    beta2 = 0.999

    class_coeff = 1e-10

    G_solver = torch.optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))
    D_solver = torch.optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2))

    class_loss = nn.CrossEntropyLoss()

    iter_count = 0

    train_history = dict(
        D_loss=[],
        G_loss=[],
        D_class_loss=[],
        G_class_loss=[]

    )
    for epoch in range(num_epochs):
        for x, y in trainval_loader:
            y = y.to(device)
            if len(x) != batch_size:
                continue

            D_solver.zero_grad()
            real_data = x

            logits_real, logits_class = D(2 * (real_data - 0.5).to(device))

            g_fake_seed = torch.randn(batch_size, 96).to(device)
            g_fake_class = torch.randint(0, num_classes, (batch_size, 1)).to(device)

            fake_images = G(g_fake_seed, g_fake_class).detach()

            logits_fake, logits_fake_class = D(fake_images.to(device))

            disc_loss = discriminator_loss(logits_real, logits_fake)
            c_loss = class_loss(logits_class, y.long()) + class_loss(logits_fake_class, g_fake_class.squeeze().long())
            d_total_error = disc_loss + class_coeff * c_loss

            d_total_error.backward()
            D_solver.step()

            G_solver.zero_grad()
            g_fake_seed = torch.randn(batch_size, 96).to(device)

            g_fake_class = torch.randint(0, num_classes, (batch_size, 1)).to(device)
            fake_images = G(g_fake_seed, g_fake_class)

            gen_logits_fake, gen_class_fake = D(fake_images)

            gen_error = generator_loss(gen_logits_fake)
            g_c_loss = class_loss(gen_class_fake, g_fake_class.squeeze().long())
            g_error = gen_error + class_coeff * g_c_loss
            g_error.backward()
            G_solver.step()

            train_history['D_loss'].append(disc_loss.item())
            train_history['G_loss'].append(gen_error.item())
            train_history['D_class_loss'].append(c_loss.item())
            train_history['G_class_loss'].append(g_c_loss.item())

            # if epoch+1 % 10==0:
            #   G_solver.param_groups[0]['lr'] /= 1.1
            #   D_solver.param_groups[0]['lr'] /= 1.1
            iter_count += 1
            if (iter_count % show_every == 0):
                print(
                    'Epoch {}, Iter: {}, D: {:.4}, G:{:.4}'.format(epoch, iter_count, d_total_error.item(), g_error.item()))
                g_fake_seed = torch.randn(batch_size, 96).to(device)
                fake_images = G(g_fake_seed, g_fake_class, flatten=False)

                a = 5
                b = 5
                fig = make_subplots(
                    rows=a, cols=b,
                    # subplot_titles=("Real pos", "Fake pos", "Real neg", "Fake neg")
                )

                places = [(i + 1, j + 1) for i in range(a) for j in range(b)]

                for i, place in enumerate(places):
                    fig.add_trace(
                        go.Heatmap(z=fake_images[i, ...].detach().cpu().squeeze(), colorscale='viridis'),
                        row=place[0], col=place[1]
                    )

                fig.update_layout(height=1000, width=500)  # width was 300
                fig.write_html(f'checkpoints_conditional/epoch_{epoch}_iter_{iter_count}.html')
                torch.save(D.state_dict(), f'mafat_D_{iter_count}.pth')
                torch.save(G.state_dict(), f'mafat_G_{iter_count}.pth')
                torch.save(embedding.state_dict(), f'mafat_embedding_{iter_count}.pth')

    g_fake_class = torch.zeros((batch_size,1)).to(device)
    g_fake_class[:15] = 1
    g_fake_seed = torch.randn(batch_size, 96).to(device)
    fake_images = G(g_fake_seed, g_fake_class, flatten=False)

    a = 5
    b = 5
    fig = make_subplots(
        rows=a, cols=b,
        # subplot_titles=("Real pos", "Fake pos", "Real neg", "Fake neg")
    )

    places = [(i + 1, j + 1) for i in range(a) for j in range(b)]

    for i, place in enumerate(places):
        fig.add_trace(
            go.Heatmap(z=fake_images[i, ...].detach().cpu().squeeze(), colorscale='viridis'),
            row=place[0], col=place[1]
        )

    fig.update_layout(height=1000, width=500)  # width was 300
    fig.show()

    # In[32]:


    plt.figure()
    x = np.arange(len(train_history['D_loss']))
    plt.plot(x, train_history['D_loss'], label='D loss')
    plt.plot(x, train_history['G_loss'], label='G loss')
    plt.xlabel('step')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid('both')
    plt.legend()
    plt.savefig('training_history.jpg')
    plt.figure()
    plt.plot(x, train_history['D_class_loss'], label='D class loss')
    plt.plot(x, train_history['G_class_loss'], label='G class loss')
    plt.xlabel('step')
    plt.ylabel('CELoss')
    plt.yscale('log')
    plt.grid('both')
    plt.legend()
    plt.savefig('training_history_class.jpg')


    torch.save(D.state_dict(), 'mafat_D.pth')
    torch.save(G.state_dict(), 'mafat_G.pth')
    torch.save(embedding.state_dict(), 'mafat_embedding.pth')


    generate_samples(G,batch_size,[64,128,256,512], 'fake_images_conditional')




