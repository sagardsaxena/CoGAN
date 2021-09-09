import torch
import torch.nn as nn


# Discriminator Model
class CoDis28x28(nn.Module):
    def __init__(self):
        super(CoDis28x28, self).__init__()
        # conv0
        self.conv0_a = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
        self.conv0_b = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
        self.bn0 = nn.BatchNorm2d(32, affine=False)
        self.prelu0 = nn.PReLU()
        # conv1
        self.conv1 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(64, affine=False)
        self.prelu1 = nn.PReLU()
        # conv2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(128, affine=False)
        self.prelu2 = nn.PReLU()
        self.drop2 = nn.Dropout(p=0.1)
        # conv3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256, affine=False)
        self.prelu3 = nn.PReLU()
        self.drop3 = nn.Dropout(p=0.3)
        # conv4
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512, affine=False)
        self.prelu4 = nn.PReLU()
        self.drop4 = nn.Dropout(p=0.3)
        # conv5
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(1024, affine=False)
        self.prelu5 = nn.PReLU()
        self.drop5 = nn.Dropout(p=0.5)
        # fc6
        self.flatten = nn.Flatten()
        self.fc6 = nn.Linear(4096, 2048) 
        self.prelu6 = nn.PReLU()
        self.drop6 = nn.Dropout(p=0.5)
        # fc7
        self.fc7 = nn.Linear(2048, 2)
        self.softmax7 = nn.Softmax(dim=1)
        # fc_cl
        self.fc_cl = nn.Linear(2048, 2)
        self.softmax_cl = nn.Softmax(dim=1)
        #self.conv0_a = nn.Conv2d(3, 20, kernel_size=5, stride=1, padding=0)
        #self.conv0_b = nn.Conv2d(3, 20, kernel_size=5, stride=1, padding=0)
        #self.pool0 = nn.MaxPool2d(kernel_size=2)
        #self.conv1 = nn.Conv2d(20, 50, kernel_size=5, stride=1, padding=0)
        #self.pool1 = nn.MaxPool2d(kernel_size=2)
        #self.conv2 = nn.Conv2d(50, 500, kernel_size=4, stride=1, padding=0)
        #self.prelu2 = nn.PReLU()
        #self.conv3 = nn.Conv2d(500, 2, kernel_size=1, stride=1, padding=0)
        #self.conv_cl = nn.Conv2d(500, 10, kernel_size=1, stride=1, padding=0)

    def forward(self, x_a, x_b):
        h0_a = self.prelu0(self.bn0(self.conv0_a(x_a)))
        h0_b = self.prelu0(self.bn0(self.conv0_b(x_b)))
        h1_a = self.prelu1(self.bn1(self.conv1(h0_a)))
        h1_b = self.prelu1(self.bn1(self.conv1(h0_b)))
        h2_a = self.drop2(self.prelu2(self.bn2(self.conv2(h1_a))))
        h2_b = self.drop2(self.prelu2(self.bn2(self.conv2(h1_b))))
        h3_a = self.drop3(self.prelu3(self.bn3(self.conv3(h2_a))))
        h3_b = self.drop3(self.prelu3(self.bn3(self.conv3(h2_b))))
        h4_a = self.drop4(self.prelu4(self.bn4(self.conv4(h3_a))))
        h4_b = self.drop4(self.prelu4(self.bn4(self.conv4(h3_b))))
        h5_a = self.drop5(self.prelu5(self.bn5(self.conv5(h4_a))))
        h5_b = self.drop5(self.prelu5(self.bn5(self.conv5(h4_b))))
        h6_a = self.drop6(self.prelu6(self.fc6(self.flatten(h5_a)))) 
        h6_b = self.drop6(self.prelu6(self.fc6(self.flatten(h5_b))))
        h6 = torch.cat((h6_a, h6_b), 0)
        h7 = self.softmax7(self.fc7(h6))
        return h7.squeeze(), h6_a, h6_b 
        #h0_a = self.pool0(self.conv0_a(x_a))
        #h0_b = self.pool0(self.conv0_b(x_b))
        #h1_a = self.pool1(self.conv1(h0_a))
        #h1_b = self.pool1(self.conv1(h0_b))
        #h2_a = self.prelu2(self.conv2(h1_a))
        #h2_b = self.prelu2(self.conv2(h1_b))
        #h2 = torch.cat((h2_a, h2_b), 0)
        #h3 = self.conv3(h2)
        #return h3.squeeze(), h2_a, h2_b

    def classify_a(self, x_a):
        h0_a = self.prelu0(self.bn0(self.conv0_a(x_a)))
        h1_a = self.prelu1(self.bn1(self.conv1(h0_a)))
        h2_a = self.drop2(self.prelu2(self.bn2(self.conv2(h1_a))))
        h3_a = self.drop3(self.prelu3(self.bn3(self.conv3(h2_a))))
        h4_a = self.drop4(self.prelu4(self.bn4(self.conv4(h3_a))))
        h5_a = self.drop5(self.prelu5(self.bn5(self.conv5(h4_a))))
        h6_a = self.drop6(self.prelu6(self.fc6(self.flatten(h5_a)))) 
        h7_a = self.softmax_cl(self.fc_cl(h6_a))
        return h7_a.squeeze()
        #h0_a = self.pool0(self.conv0_a(x_a))
        #h1_a = self.pool1(self.conv1(h0_a))
        #h2_a = self.prelu2(self.conv2(h1_a))
        #h3_a = self.conv_cl(h2_a)
        #return h3_a.squeeze()

    def classify_b(self, x_b):
        h0_b = self.prelu0(self.bn0(self.conv0_b(x_b)))
        h1_b = self.prelu1(self.bn1(self.conv1(h0_b)))
        h2_b = self.drop2(self.prelu2(self.bn2(self.conv2(h1_b))))
        h3_b = self.drop3(self.prelu3(self.bn3(self.conv3(h2_b))))
        h4_b = self.drop4(self.prelu4(self.bn4(self.conv4(h3_b))))
        h5_b = self.drop5(self.prelu5(self.bn5(self.conv5(h4_b))))
        h6_b = self.drop6(self.prelu6(self.fc6(self.flatten(h5_b))))
        h7_b = self.softmax_cl(self.fc_cl(h6_b))
        return h7_b.squeeze() 
        #h0_b = self.pool0(self.conv0_b(x_b))
        #h1_b = self.pool1(self.conv1(h0_b))
        #h2_b = self.prelu2(self.conv2(h1_b))
        #h3_b = self.conv_cl(h2_b)
        #return h3_b.squeeze()

# Generator Model
class CoGen28x28(nn.Module):
    def __init__(self, latent_dims):
        super(CoGen28x28, self).__init__()
        # dconv0
        self.dconv0 = nn.ConvTranspose2d(latent_dims, 1024, kernel_size=4, stride=1)
        self.bn0 = nn.BatchNorm2d(1024, affine=False)
        self.prelu0 = nn.PReLU()
        # dconv1
        self.dconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(512, affine=False)
        self.prelu1 = nn.PReLU()
        # dconv2
        self.dconv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256, affine=False)
        self.prelu2 = nn.PReLU()
        # dconv3
        self.dconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128, affine=False)
        self.prelu3 = nn.PReLU()
        # dconv4
        self.dconv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(64, affine=False)
        self.prelu4 = nn.PReLU()
        # dconv5
        self.dconv5 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(32, affine=False)
        self.prelu5 = nn.PReLU()
        # dconv6
        self.dconv6_a = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1)
        self.dconv6_b = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1)
        self.tanh6_a = nn.Tanh()
        self.tanh6_b = nn.Tanh()
        
        #self.dconv0 = nn.ConvTranspose2d(latent_dims, 1024, kernel_size=4, stride=1)
        #self.bn0 = nn.BatchNorm2d(1024, affine=False)
        #self.prelu0 = nn.PReLU()
        #self.dconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1)
        #self.bn1 = nn.BatchNorm2d(512, affine=False)
        #self.prelu1 = nn.PReLU()
        #self.dconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1)
        #self.bn2 = nn.BatchNorm2d(256, affine=False)
        #self.prelu2 = nn.PReLU()
        #self.dconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1)
        #self.bn3 = nn.BatchNorm2d(128, affine=False)
        #self.prelu3 = nn.PReLU()
        #self.dconv4_a = nn.ConvTranspose2d(128, 3, kernel_size=6, stride=1, padding=1)
        #self.dconv4_b = nn.ConvTranspose2d(128, 3, kernel_size=6, stride=1, padding=1)
        #self.sig4_a = nn.Sigmoid()
        #self.sig4_b = nn.Sigmoid()

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        h0 = self.prelu0(self.bn0(self.dconv0(z)))
        h1 = self.prelu1(self.bn1(self.dconv1(h0)))
        h2 = self.prelu2(self.bn2(self.dconv2(h1)))
        h3 = self.prelu3(self.bn3(self.dconv3(h2)))
        h4 = self.prelu4(self.bn4(self.dconv4(h3)))
        h5 = self.prelu5(self.bn5(self.dconv5(h4)))
        out_a = self.tanh6_a(self.dconv6_a(h5))
        out_b = self.tanh6_b(self.dconv6_b(h5))
        return out_a, out_b
        #z = z.view(z.size(0), z.size(1), 1, 1)
        #h0 = self.prelu0(self.bn0(self.dconv0(z)))
        #h1 = self.prelu1(self.bn1(self.dconv1(h0)))
        #h2 = self.prelu2(self.bn2(self.dconv2(h1)))
        #h3 = self.prelu3(self.bn3(self.dconv3(h2)))
        #out_a = self.sig4_a(self.dconv4_a(h3))
        #out_b = self.sig4_b(self.dconv4_b(h3))
        #return out_a, out_b

