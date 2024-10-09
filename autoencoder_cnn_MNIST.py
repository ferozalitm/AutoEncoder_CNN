#  autoencoder_cnn_MNIST.py
#     Autoencoder using 3 layer conv network all RelU except last reconstruction Sigmoid layer, 10 dimn Latent and BCE on reconstruction data
#     (N/w:
#           Encoder ch: (3->32->64-> fc layer (64X3X3 -> 9))
#                   img: 28 -> 11 -> 3 -> fc
#                       
#           Decoder ch: (3->32->64-> fc layer (64X3X3 -> 9))
#                   img: fc -> 3 -> 12 -> 28 
#
#       Interpolate between digits using the latent space.
#

import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import random
from torchvision.utils import save_image
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
from sklearn.decomposition import PCA
import matplotlib.cm as cm

no_classes = 10
colors = cm.rainbow(np.linspace(0, 1, no_classes))

# Reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Hyperparameters
ip_dimn = 28*28
no_classes = 10
batch_size = 256
latent_dimn = 9

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Create a directory if not exists
sample_dir = './Autoencoder_conv_results_wMaxPool_interpolate/10dimnLatentSpace/BCE_loss'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

train_dataset = torchvision.datasets.MNIST(root='../data/',
                                     train=True, 
                                     transform=transforms.ToTensor(),
                                     download=True)

test_dataset = torchvision.datasets.MNIST(root='../data/',
                                     train=False, 
                                     transform=transforms.ToTensor(),
                                     download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size, 
                                           shuffle=False)

# Finding image size and channel depth
# train_dataset[0][0].shape  -> torch.Size([3, 32, 32])
ip_image_size = train_dataset[0][0].shape[1]
print(f'image_size: {ip_image_size}')
ip_image_ch = train_dataset[0][0].shape[0]
print(f'ip_image_ch: {ip_image_ch}')
print(ip_image_ch)

no_batches_train = len(train_loader)
no_batches_tst = len(test_loader)
print(f"No_batches train: {no_batches_train}")
print(f"No_batches test: {no_batches_tst}")

# Build a fully connected layer and forward pass
class AutoEncoderConvNet(nn.Module):
    def __init__(self, ip_image_size):
        super().__init__()
        self.conv_op_image_size_encoder_layer1 = ip_image_size

        # Encoder layers
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride=2))
        self.conv_op_image_size_encoder_layer1 = (self.conv_op_image_size_encoder_layer1 - 5)//1 + 1        # Conv -> (24X24)
        self.conv_op_image_size_encoder_layer1 = (self.conv_op_image_size_encoder_layer1 - 3)//2 + 1        # Conv -> (11X11)

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride=2))
        self.conv_op_image_size_encoder_layer1 = (self.conv_op_image_size_encoder_layer1 - 5)//1 + 1        # Conv -> (7X7)
        self.conv_op_image_size_encoder_layer2 = (self.conv_op_image_size_encoder_layer1 - 3)//2 + 1        # Conv -> (3X3)
        
        # print(self.conv_op_image_size_encoder)
        self.fc_layer_size = self.conv_op_image_size_encoder_layer2*self.conv_op_image_size_encoder_layer2*64
        print(f'conv_op_image_size_encoder:{self.conv_op_image_size_encoder_layer2}, fc_layer_size:{self.fc_layer_size}')
        
        self.fc = nn.Linear(in_features=self.fc_layer_size, out_features=latent_dimn)      


        # Decoder layers
        self.fc_d = nn.Linear(in_features=latent_dimn, out_features=self.fc_layer_size)          

        self.layer2_d = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size = 5, stride=1),     # Transpose -> (7X7)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),                       # Bilinear -> (14X14)
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride=1),              # Conv -> (12X12)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding=1, stride=1),   # Conv -> (12X12)
            nn.BatchNorm2d(32),
            nn.ReLU()
            )
        self.conv_op_image_size_decoder_layer2 = self.conv_op_image_size_encoder_layer2 + 5 - 1     
        self.conv_op_image_size_decoder_layer2 = self.conv_op_image_size_decoder_layer2*2           
        self.conv_op_image_size_decoder_layer2 = self.conv_op_image_size_encoder_layer2 - 3 + 1     

        self.layer1_d = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 32, out_channels = 1, kernel_size = 5, stride=1),      # Transpose -> (16X16)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),                       # Bilinear -> (32X32)
            nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 3, stride=1),                # Conv -> (30X30)
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 3, stride=1),                # Conv -> (28X28)
            nn.BatchNorm2d(1),
            nn.Sigmoid())
        self.conv_op_image_size_decoder_layer1 = self.conv_op_image_size_encoder_layer2 + 5 - 1     # Conv -> (16X16)
        self.conv_op_image_size_decoder_layer1 = self.conv_op_image_size_decoder_layer1 *2          # Transpose (32X32)
        self.conv_op_image_size_decoder_layer1 = self.conv_op_image_size_decoder_layer1 - 3 + 1     # Transpose (30X30)
        self.conv_op_image_size_decoder_layer1 = self.conv_op_image_size_decoder_layer1 - 3 + 1     # Transpose (28X28)
        

    def forward(self, x):
        e = self.Encoder(x)
        d = self.Decoder(e)
        return e,d
    
    def Encoder(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(-1, self.fc_layer_size )
        x = self.fc(x)
        return x

    def Decoder(self, x):
        # breakpoint()
        x = self.fc_d(x)
        x = x.reshape(-1, 64, self.conv_op_image_size_encoder_layer2, self.conv_op_image_size_encoder_layer2 )
        x = self.layer2_d(x)
        x = self.layer1_d(x)
        return x
        

# Build model.
model = AutoEncoderConvNet(ip_image_size).to(device)

# Build optimizer.
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# For updating learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Build loss.
# criterion = nn.MSELoss()
criterion = nn.BCELoss()

no_epochs = 150
first_pass = True
epoch_all = []
loss_test_all = []
loss_train_all = []

curr_lr = learning_rate
for epoch in range(no_epochs):

  # Training
  batch_idx = 0
  total_loss_train = 0

  for batch_idx, (images, labels) in enumerate(train_loader):

    images = images.to(device)
    # labels = labels.to(device)

    # Forward pass.
    latent_encoding, x_reconst = model(images)

    # Compute loss.
    loss = criterion(x_reconst, images)

    if epoch == 0 and first_pass == True:
      print(f'Initial {epoch} loss: ', loss.item())
      first_pass = False

    # Compute gradients.
    optimizer.zero_grad()
    loss.backward()

    # 1-step gradient descent.
    optimizer.step()

    # calculating train loss
    total_loss_train += loss.item()

    if epoch == 0 and (batch_idx+1) % 10 == 0:
      print(f"Train Batch:{batch_idx}/{no_batches_train}, loss: {loss}, total_loss: {total_loss_train}")

    if (epoch % 5 == 0) or (epoch == no_epochs-1):
        # Accumulate data for PCA
        if batch_idx == 0:
            # X_train = latent_encoding.detach().cpu().numpy()
            X_train = latent_encoding.detach()
            X_labels = labels.detach().cpu().numpy()
        else:
            # X_train = np.concatenate((X_train, latent_encoding.detach().cpu().numpy()), axis=0)
            X_train = torch.cat((X_train, latent_encoding.detach()), axis=0)
            X_labels = np.concatenate((X_labels, labels.detach().cpu().numpy()), axis=0)

  # Decay learning rate
  if (epoch+1) % 50 == 0:
      curr_lr /= 10
      update_lr(optimizer, curr_lr)

  print(f'Train Epoch:{epoch}, Average Train loss:{total_loss_train/no_batches_train}' )

  if (epoch % 5 == 0) or (epoch == no_epochs-1):
    x_concat = torch.cat([images.view(-1, 1, 28, 28), x_reconst.view(-1, 1, 28, 28)], dim=3)
    save_image(x_concat, os.path.join(sample_dir, 'train_reconst-{}.png'.format(epoch)))

  # Testing after each epoch
  model.eval()
  with torch.no_grad():

    total_loss_test = 0

    for images, labels in test_loader:

      images = images.to(device)
      # labels = labels.to(device)

      # Forward pass.
      _, x_reconst = model(images)

      # Compute test loss.
      loss = criterion(x_reconst, images)
      
      total_loss_test += loss.item()

    print(f'Test Epoch:{epoch}, Average Test loss: {total_loss_test/no_batches_tst}')

    if (epoch % 5 == 0) or (epoch == no_epochs-1):
      x_concat = torch.cat([images.view(-1, 1, 28, 28), x_reconst.view(-1, 1, 28, 28)], dim=3)
      save_image(x_concat, os.path.join(sample_dir, 'test_reconst-{}.png'.format(epoch)))

  # PLotting train and test curves
  epoch_all.append(epoch)
  loss_test_all.append(total_loss_test/no_batches_tst)
  loss_train_all.append(total_loss_train/no_batches_train)

  plt.clf()
  plt.plot(epoch_all, loss_train_all, marker = 'o', mec = 'g', label='Average Train loss')
  plt.plot(epoch_all, loss_test_all, marker = 'o', mec = 'r', label='Average Test loss')
  plt.legend()
  plt.title('Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.show()
  plt.savefig(os.path.join(sample_dir, 'Loss.png'))

  # Plotting latent space using PCA
  if (epoch % 5 == 0) or (epoch == no_epochs-1):

    X_trainT = sc.fit_transform(X_train.cpu().numpy())
    # X_test = sc.transform(X_test)
    pca = PCA(n_components = 2)
    X_trainPCA = pca.fit_transform(X_trainT)
    # X_test = pca.transform(X_test)
    # breakpoint()

    plt.clf()
    no_points_plt = 10000
    X = X_trainPCA[0:no_points_plt,0]
    Y = X_trainPCA[0:no_points_plt,1]
    plt.scatter(X, Y, color = colors[X_labels[0:no_points_plt]])
    plt.title('PCA latent space')
    plt.show()
    plt.savefig(os.path.join(sample_dir, f'PCA_latentSpace-{epoch}.png'))

  # Interpolating latent space of digits
  if (epoch % 5 == 0) or (epoch == no_epochs-1):

    # Interpolating latent space of digits
    X_labels_1_idx = (X_labels == 1).nonzero()
    X_encode_1 = X_train[X_labels_1_idx]

    X_labels_7_idx = (X_labels == 7).nonzero()
    X_encode_7 = X_train[X_labels_7_idx]

    X_labels_3_idx = (X_labels == 3).nonzero()
    X_encode_3 = X_train[X_labels_3_idx]

    X_labels_8_idx = (X_labels == 8).nonzero()
    X_encode_8 = X_train[X_labels_8_idx]

    X_encode_1_mean = torch.mean(X_encode_1, axis=0)
    X_encode_7_mean = torch.mean(X_encode_7, axis=0)
    X_encode_3_mean = torch.mean(X_encode_3, axis=0)
    X_encode_8_mean = torch.mean(X_encode_8, axis=0)
    X_encode_1_mean = X_encode_1_mean[None,:]
    X_encode_7_mean = X_encode_7_mean[None,:]
    X_encode_3_mean = X_encode_3_mean[None,:]
    X_encode_8_mean = X_encode_8_mean[None,:]

    # breakpoint()
    linear_scale = np.linspace(0, 1, no_classes)
    X_latent_samples_1_7_concat = X_encode_1_mean
    X_latent_samples_3_8_concat = X_encode_3_mean

    for idx in range(1,10):        
        # 1D interpolation
        X_latent_samples_1_7 = X_encode_1_mean*(1-linear_scale[idx]) + X_encode_7_mean*linear_scale[idx]
        X_latent_samples_1_7_concat = torch.cat([X_latent_samples_1_7_concat, X_latent_samples_1_7], dim=0)

        X_latent_samples_3_8 = X_encode_3_mean*(1-linear_scale[idx]) + X_encode_8_mean*linear_scale[idx]
        X_latent_samples_3_8_concat = torch.cat([X_latent_samples_3_8_concat, X_latent_samples_3_8], dim=0)

    X_image_samples_1_7 = model.Decoder(X_latent_samples_1_7_concat)
    X_image_samples_3_8 = model.Decoder(X_latent_samples_3_8_concat)

    save_image(X_image_samples_1_7, os.path.join(sample_dir, 'sampled_1_7-{}.png'.format(epoch)), nrow = 10)
    save_image(X_image_samples_3_8, os.path.join(sample_dir, 'sampled_3_8-{}.png'.format(epoch)), nrow = 10)

    # 2D interpolation
    for idx2 in range(0,10):   
        for idx1 in range(0,10):         

          if idx2 == 0 and idx1 == 0:
              X_latent_samples_1_7_3_8_concat = X_encode_1_mean
          else:
            X_latent_samples_1_7 = X_encode_1_mean*(1-linear_scale[idx1]) + X_encode_7_mean*linear_scale[idx1]
            X_latent_samples_3_8 = X_encode_3_mean*(1-linear_scale[idx1]) + X_encode_8_mean*linear_scale[idx1]
            X_latent_samples_1_7_3_8 = X_latent_samples_1_7*(1-linear_scale[idx2]) + X_latent_samples_3_8*linear_scale[idx2]
            X_latent_samples_1_7_3_8_concat = torch.cat([X_latent_samples_1_7_3_8_concat, X_latent_samples_1_7_3_8], dim=0)

    X_image_samples_1_7_3_8 = model.Decoder(X_latent_samples_1_7_3_8_concat)

    save_image(X_image_samples_1_7_3_8, os.path.join(sample_dir, 'sampled_1_7_3_8-{}.png'.format(epoch)), nrow = 10)

  model.train()
