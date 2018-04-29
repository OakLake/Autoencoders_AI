# Pytorch Autoencoder

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.utils import save_image


def mnist_data():
    compose = transforms.Compose([transforms.ToTensor(),transforms.Normalize((.5,.5,.5),(.5,.5,.5))])
    out_dir = "./dataset"
    return datasets.MNIST(root=out_dir,train=True,transform=compose,download=True)


# Load MNIST dataset
data = mnist_data()

# Create loader with data
data_loader = torch.utils.data.DataLoader(data,batch_size=100,shuffle=True)

# Num_batches
num_batches = len(data_loader)

# Autoencoder Network
class VAE(nn.Module):
    """ Variational AutoEncoder Network"""
    def __init__(self):
        super(VAE,self).__init__()
        n_features = 784
        n_out = 784

        self.encoder = nn.Sequential(
        nn.Linear(n_features,256),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.3),
        nn.Linear(256,128),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.3),
        nn.Linear(128,64),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.3),
        nn.Linear(64,32),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.3)
        )
        self.decoder = nn.Sequential(
        nn.Linear(32,64),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.3),
        nn.Linear(64,128),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.3),
        nn.Linear(128,256),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.3),
        nn.Linear(256,n_out),
        nn.Tanh(),
        nn.Dropout(0.3)
        )

    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

vae_net = VAE()

def images_to_vectors(images):
    return images.view(images.size(0),784)

def vectors_to_images(vectors):
    return vectors.view(vectors.size(0),1,28,28)


loss = nn.MSELoss()
optimizer = optim.Adam(vae_net.parameters(),lr=0.0002)


num_epochs = 1000
for epoch in range(num_epochs):
    for data in data_loader:
        img,_ = data
        img = images_to_vectors(img)
        img = Variable(img)
        ## === Forward Pass
        output = vae_net(img)
        loss_ = loss(output,img)
        ## == Backward Pass
        optimizer.zero_grad()
        loss_.backward()
        optimizer.step()

    ## == Log
    print("Epoch [{}/{}]  loss: {}".format(epoch,num_epochs,loss_.data[0]))
    if epoch%10 == 0:
        pic = vectors_to_images(output.data)
        save_image(pic,'./VAE_IMGs/vae_img_{}.png'.format(epoch),nrow=20)

torch.save(model.state_dict(), './sim_autoencoder.pth')





# eop
