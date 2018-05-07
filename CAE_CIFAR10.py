# Pytorch Convolutional Autoencoder

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.utils import save_image
import time

print('PyTorch version: ',torch.__version__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device: ',device)

def cifar10_data():
    compose = transforms.Compose([transforms.ToTensor(),transforms.Normalize((.5,.5,.5),(.5,.5,.5))])
    out_dir = "./dataset"
    return datasets.CIFAR10(root=out_dir,train=True,transform=compose,download=True)


# Load MNIST dataset
data = cifar10_data()

# Create loader with data
data_loader = torch.utils.data.DataLoader(data,batch_size=32,shuffle=True)

# Num_batches
num_batches = len(data_loader)

# Autoencoder Network
class CAE(nn.Module):
    """ Variational AutoEncoder Network"""
    def __init__(self):
        super(CAE,self).__init__()

        self.encoder = nn.Sequential(
            torch.nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            torch.nn.Conv2d(6, 12, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            torch.nn.Conv2d(12, 24, kernel_size=3, stride=1, padding=1)

        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(24, 12, kernel_size=7, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(12, 8, kernel_size=7, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(8, 6, kernel_size=9, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(6, 3, kernel_size=13, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

cae_net = CAE()
cae_net.to(device) # move to GPU



loss = nn.MSELoss()
optimizer = optim.Adam(cae_net.parameters(),lr=0.0002)

print('Training...')
num_epochs = 1000
for epoch in range(num_epochs):
    epoch_time = time.time()
    for data in data_loader:
        img,_ = data
        #img = images_to_vectors(img)
        img = Variable(img).to(device) # move to GPU
        ## === Forward Pass
        output = cae_net(img)
        loss_ = loss(output,img)
        ## == Backward Pass
        optimizer.zero_grad()
        loss_.backward()
        optimizer.step()


    ## == Log
    print("Epoch [{}/{}]  loss: {}  :: epoch duration {}".format(epoch,num_epochs,loss_.data[0], (time.time()-epoch_time) ))
    # if epoch%10 == 0:
    pic = output.data#vectors_to_images(output.data)
    save_image(pic,'./CAE_CIFAR10_IMGs/cae_img_{}.png'.format(epoch),nrow=20)

torch.save(cae_net.state_dict(), './sim_autoencoder_cifar10_cae.pth')





# eop
