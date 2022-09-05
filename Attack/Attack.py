# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# + [markdown] id="Q-n2e0BkhEKS"
# # **Homework 10 - Adversarial Attack**
#
# Slides: https://reurl.cc/7DDxnD
#
# Contact: ntu-ml-2022spring-ta@googlegroups.com
#

# + [markdown] id="9RX7iRXrhMA_"
# ## Enviroment & Download
#
# We make use of [pytorchcv](https://pypi.org/project/pytorchcv/) to obtain CIFAR-10 pretrained model, so we need to set up the enviroment first. We also need to download the data (200 images) which we want to attack.

# + id="d4Lw7urignqP" colab={"base_uri": "https://localhost:8080/"} outputId="32ec255b-3b9b-43b8-c318-e04d1009f0fd"
# set up environment
# !pip install pytorchcv
# !pip install imgaug

# download
# !wget https://github.com/DanielLin94144/ML-attack-dataset/files/8167812/data.zip

# unzip
# !unzip ./data.zip
# !rm ./data.zip

# + id="5inbFx_alYjw"
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 8

# + [markdown] id="hkQQf0l1hbBs"
# ## Global Settings 
# #### **[NOTE]**: Don't change the settings here, or your generated image might not meet the constraint.
# * $\epsilon$ is fixed to be 8. But on **Data section**, we will first apply transforms on raw pixel value (0-255 scale) **by ToTensor (to 0-1 scale)** and then **Normalize (subtract mean divide std)**. $\epsilon$ should be set to $\frac{8}{255 * std}$ during attack.
#
# * Explaination (optional)
#     * Denote the first pixel of original image as $p$, and the first pixel of adversarial image as $a$.
#     * The $\epsilon$ constraints tell us $\left| p-a \right| <= 8$.
#     * ToTensor() can be seen as a function where $T(x) = x/255$.
#     * Normalize() can be seen as a function where $N(x) = (x-mean)/std$ where $mean$ and $std$ are constants.
#     * After applying ToTensor() and Normalize() on $p$ and $a$, the constraint becomes $\left| N(T(p))-N(T(a)) \right| = \left| \frac{\frac{p}{255}-mean}{std}-\frac{\frac{a}{255}-mean}{std} \right| = \frac{1}{255 * std} \left| p-a \right| <= \frac{8}{255 * std}.$
#     * So, we should set $\epsilon$ to be $\frac{8}{255 * std}$ after ToTensor() and Normalize().

# + id="ACghc_tsg2vE"
# the mean and std are the calculated statistics from cifar_10 dataset
cifar_10_mean = (0.491, 0.482, 0.447) # mean for the three channels of cifar_10 images
cifar_10_std = (0.202, 0.199, 0.201) # std for the three channels of cifar_10 images

# convert mean and std to 3-dimensional tensors for future operations
mean = torch.tensor(cifar_10_mean).to(device).view(3, 1, 1)
std = torch.tensor(cifar_10_std).to(device).view(3, 1, 1)

epsilon = 8/255/std

# + id="uO8f0NmtlM63"
root = './data' # directory for storing benign images
# benign images: images which do not contain adversarial perturbations
# adversarial images: images which include adversarial perturbations

# + [markdown] id="lhBJBAlKherZ"
# ## Data
#
# Construct dataset and dataloader from root directory. Note that we store the filename of each image for future usage.

# + colab={"base_uri": "https://localhost:8080/"} id="VXpRAHz0hkDt" outputId="5cc1aa38-d561-4fc1-d891-600313ddf87f"
import os
import glob
import shutil
import numpy as np
from PIL import Image
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar_10_mean, cifar_10_std)
])

class AdvDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.images = []
        self.labels = []
        self.names = []
        '''
        data_dir
        ├── class_dir
        │   ├── class1.png
        │   ├── ...
        │   ├── class20.png
        '''
        for i, class_dir in enumerate(sorted(glob.glob(f'{data_dir}/*'))):
            images = sorted(glob.glob(f'{class_dir}/*'))
            self.images += images
            self.labels += ([i] * len(images))
            self.names += [os.path.relpath(imgs, data_dir) for imgs in images]
        self.transform = transform
    def __getitem__(self, idx):
        image = self.transform(Image.open(self.images[idx]))
        label = self.labels[idx]
        return image, label
    def __getname__(self):
        return self.names
    def __len__(self):
        return len(self.images)

adv_set = AdvDataset(root, transform=transform)
adv_names = adv_set.__getname__()
adv_loader = DataLoader(adv_set, batch_size=batch_size, shuffle=False)

print(f'number of images = {adv_set.__len__()}')


# + [markdown] id="LnszlTsYrTQZ"
# ## Utils -- Benign Images Evaluation

# + id="5c_zZLzkrceE"
# to evaluate the performance of model on benign images
def epoch_benign(model, loader, loss_fn):
    model.eval()
    train_acc, train_loss = 0.0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        yp = model(x)
        loss = loss_fn(yp, y)
        train_acc += (yp.argmax(dim=1) == y).sum().item()
        train_loss += loss.item() * x.shape[0]
    return train_acc / len(loader.dataset), train_loss / len(loader.dataset)


# + [markdown] id="_YJxK7YehqQy"
# ## Utils -- Attack Algorithm

# + id="F_1wKfKyhrQW"
# perform fgsm attack
def fgsm(model, x, y, loss_fn, epsilon=epsilon):
    x_adv = x.detach().clone() # initialize x_adv as original benign image x
    x_adv.requires_grad = True # need to obtain gradient of x_adv, thus set required grad
    loss = loss_fn(model(x_adv), y) # calculate loss
    loss.backward() # calculate gradient
    # fgsm: use gradient ascent on x_adv to maximize loss
    grad = x_adv.grad.detach()
    x_adv = x_adv + epsilon * grad.sign()
    return x_adv

# alpha and num_iter can be decided by yourself
alpha = 0.8/255/std
def ifgsm(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=20):
    x_adv = x
    # write a loop of num_iter to represent the iterative times
    for i in range(num_iter):
        # x_adv = fgsm(model, x_adv, y, loss_fn, alpha) # call fgsm with (epsilon = alpha) to obtain new x_adv
        x_adv = x_adv.detach().clone()
        x_adv.requires_grad = True # need to obtain gradient of x_adv, thus set required grad
        loss = loss_fn(model(x_adv), y) # calculate loss
        loss.backward() # calculate gradient
        # fgsm: use gradient ascent on x_adv to maximize loss
        grad = x_adv.grad.detach()
        x_adv = x_adv + alpha * grad.sign()

        x_adv = torch.max(torch.min(x_adv, x+epsilon), x-epsilon) # clip new x_adv back to [x-epsilon, x+epsilon]
    return x_adv

def mifgsm(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=20, decay=1.0):
    x_adv = x
    # initialze momentum tensor
    momentum = torch.zeros_like(x).detach().to(device)
    # write a loop of num_iter to represent the iterative times
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        x_adv.requires_grad = True # need to obtain gradient of x_adv, thus set required grad
        loss = loss_fn(model(x_adv), y) # calculate loss
        loss.backward() # calculate gradient
        # TODO: Momentum calculation
        # grad = .....
        x_adv = x_adv + alpha * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x+epsilon), x-epsilon) # clip new x_adv back to [x-epsilon, x+epsilon]
    return x_adv


# + [markdown] id="fYCEQwmcrmH6"
# ## Utils -- Attack
# * Recall
#   * ToTensor() can be seen as a function where $T(x) = x/255$.
#   * Normalize() can be seen as a function where $N(x) = (x-mean)/std$ where $mean$ and $std$ are constants.
#
# * Inverse function
#   * Inverse Normalize() can be seen as a function where $N^{-1}(x) = x*std+mean$ where $mean$ and $std$ are constants.
#   * Inverse ToTensor() can be seen as a function where $T^{-1}(x) = x*255$.
#
# * Special Noted
#   * ToTensor() will also convert the image from shape (height, width, channel) to shape (channel, height, width), so we also need to transpose the shape back to original shape.
#   * Since our dataloader samples a batch of data, what we need here is to transpose **(batch_size, channel, height, width)** back to **(batch_size, height, width, channel)** using np.transpose.

# + id="w5X_9x-7ro_w"
# perform adversarial attack and generate adversarial examples
def gen_adv_examples(model, loader, attack, loss_fn):
    model.eval()
    adv_names = []
    train_acc, train_loss = 0.0, 0.0
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        x_adv = attack(model, x, y, loss_fn) # obtain adversarial examples
        yp = model(x_adv)
        loss = loss_fn(yp, y)
        train_acc += (yp.argmax(dim=1) == y).sum().item()
        train_loss += loss.item() * x.shape[0]
        # store adversarial examples
        adv_ex = ((x_adv) * std + mean).clamp(0, 1) # to 0-1 scale
        adv_ex = (adv_ex * 255).clamp(0, 255) # 0-255 scale
        adv_ex = adv_ex.detach().cpu().data.numpy().round() # round to remove decimal part
        adv_ex = adv_ex.transpose((0, 2, 3, 1)) # transpose (bs, C, H, W) back to (bs, H, W, C)
        adv_examples = adv_ex if i == 0 else np.r_[adv_examples, adv_ex]
    return adv_examples, train_acc / len(loader.dataset), train_loss / len(loader.dataset)

# create directory which stores adversarial examples
def create_dir(data_dir, adv_dir, adv_examples, adv_names):
    if os.path.exists(adv_dir) is not True:
        _ = shutil.copytree(data_dir, adv_dir)
    for example, name in zip(adv_examples, adv_names):
        im = Image.fromarray(example.astype(np.uint8)) # image pixel value should be unsigned int
        im.save(os.path.join(adv_dir, name))


# + [markdown] id="r_pMkmPytX3k"
# ## Model / Loss Function
#
# Model list is available [here](https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/model_provider.py). Please select models which has _cifar10 suffix. Some of the models cannot be accessed/loaded. You can safely skip them since TA's model will not use those kinds of models.

# + id="gJcKiQNUgnPQ"
from pytorchcv.model_provider import get_model as ptcv_get_model

class ensembleNet(nn.Module):
    def __init__(self, model_names):
        super(ensembleNet, self).__init__()
        self.model1 = ptcv_get_model(model_names[0], pretrained=True)
        self.model2 = ptcv_get_model(model_names[1], pretrained=True)
        self.model3 = ptcv_get_model(model_names[2], pretrained=True)
        self.model4 = ptcv_get_model(model_names[3], pretrained=True)
        self.model5 = ptcv_get_model(model_names[4], pretrained=True)
        self.model6 = ptcv_get_model(model_names[5], pretrained=True)
        self.model7 = ptcv_get_model(model_names[6], pretrained=True)
        self.model8 = ptcv_get_model(model_names[7], pretrained=True)
        self.model9 = ptcv_get_model(model_names[8], pretrained=True)
        self.model10 = ptcv_get_model(model_names[9], pretrained=True)
        self.model11 = ptcv_get_model(model_names[10], pretrained=True)
        self.model12 = ptcv_get_model(model_names[11], pretrained=True)
        self.model13 = ptcv_get_model(model_names[12], pretrained=True)
        self.model14 = ptcv_get_model(model_names[13], pretrained=True)
        self.model15 = ptcv_get_model(model_names[14], pretrained=True)
        self.model16 = ptcv_get_model(model_names[15], pretrained=True)
        self.model17 = ptcv_get_model(model_names[16], pretrained=True)
        
    def forward(self, x):
        #for i, m in enumerate(self.models):
        # TODO: sum up logits from multiple models  
        x1 = self.model1(x.clone())
        x2 = self.model2(x.clone())
        x3 = self.model3(x.clone())
        x4 = self.model4(x.clone())
        x5 = self.model5(x.clone())
        x6 = self.model6(x.clone())
        x7 = self.model7(x.clone())
        x8 = self.model8(x.clone())
        x9 = self.model9(x.clone())
        x10 = self.model10(x.clone())
        x11 = self.model11(x.clone())
        x12 = self.model12(x.clone())
        x13 = self.model13(x.clone())
        x14 = self.model14(x.clone())
        x15 = self.model15(x.clone())
        x16 = self.model16(x.clone())
        x17 = self.model17(x.clone())

        x = (x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15+x16+x17)/ 17
        return x


# + id="stYFytogeIzI"
model_names = [
      'resnext29_16x64d_cifar10',
      'resnext29_32x4d_cifar10',
      'resnet56_cifar10',
      'resnet110_cifar10',
      'resnet1001_cifar10',   
      'resnet1202_cifar10',
      'preresnet56_cifar10',
      'preresnet110_cifar10',
      'preresnet164bn_cifar10',
      #'seresnet56_cifar10',
      'seresnet110_cifar10',
      'sepreresnet56_cifar10',
      'sepreresnet110_cifar10',
      #'sepreresnet164bn_cifar10',
      'diaresnet56_cifar10',
      'diaresnet110_cifar10',
      'nin_cifar10',
      #'diapreresnet1202_cifar10',
      #'resnet164bn_cifar10',
      #'resnet272bn_cifar10',
      'diapreresnet56_cifar10',      
      'diapreresnet110_cifar10'
]
model = ensembleNet(model_names).to(device)
model.eval()

# + colab={"base_uri": "https://localhost:8080/"} id="jwto8xbPtYzQ" outputId="3f9d77c1-4077-4b67-a3fa-8c69553dfaa1"


#model = ptcv_get_model('resnet110_cifar10', pretrained=True).to(device)
loss_fn = nn.CrossEntropyLoss()

benign_acc, benign_loss = epoch_benign(model, adv_loader, loss_fn)
print(f'benign_acc = {benign_acc:.5f}, benign_loss = {benign_loss:.5f}')

# + [markdown] id="uslb7GPchtMI"
# ## FGSM

# + colab={"base_uri": "https://localhost:8080/"} id="wQwPTVUIhuTS" outputId="cc000139-7440-4b2e-aeef-bc30b9d94e9a"
adv_examples, fgsm_acc, fgsm_loss = gen_adv_examples(model, adv_loader, fgsm, loss_fn)
print(f'fgsm_acc = {fgsm_acc:.5f}, fgsm_loss = {fgsm_loss:.5f}')

create_dir(root, 'fgsm', adv_examples, adv_names)

# + [markdown] id="WXw6p0A6shZm"
# ## I-FGSM

# + colab={"base_uri": "https://localhost:8080/"} id="fUEsT06Iskt2" outputId="712d6ec3-3553-4685-bd44-a565a4f4e6ac"
adv_examples, ifgsm_acc, ifgsm_loss = gen_adv_examples(model, adv_loader, ifgsm, loss_fn)
print(f'ifgsm_acc = {ifgsm_acc:.5f}, ifgsm_loss = {ifgsm_loss:.5f}')

create_dir(root, 'ifgsm', adv_examples, adv_names)

# + [markdown] id="DQ-nYkkYexEE"
# ## Compress the images
# * Submit the .tgz file to [JudgeBoi](https://ml.ee.ntu.edu.tw/hw10/)

# + colab={"base_uri": "https://localhost:8080/"} id="ItRo_S0M264N" outputId="17757366-5b91-4aaf-b579-225d3126b962"
# %cd fgsm
# !tar zcvf ../fgsm.tgz *
# %cd ..

# %cd ifgsm
# !tar zcvf ../ifgsm.tgz *
# %cd ..

# + [markdown] id="WLZLbebigCA2"
# ## Example of Ensemble Attack
# * Ensemble multiple models as your proxy model to increase the black-box transferability ([paper](https://arxiv.org/abs/1611.02770))

# + [markdown] id="yjfJwJKeeaR2"
# * Construct your ensemble model

# + [markdown] id="0FM_S886kFd8"
# ## Visualization

# + colab={"base_uri": "https://localhost:8080/", "height": 735} id="2FCuE2njkH1O" outputId="2732ca0e-d40c-4799-c8f1-b047659a4ecd"
import matplotlib.pyplot as plt

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10, 20))
cnt = 0
for i, cls_name in enumerate(classes):
    path = f'{cls_name}/{cls_name}1.png'
    # benign image
    cnt += 1
    plt.subplot(len(classes), 4, cnt)
    im = Image.open(f'./data/{path}')
    logit = model(transform(im).unsqueeze(0).to(device))[0]
    predict = logit.argmax(-1).item()
    prob = logit.softmax(-1)[predict].item()
    plt.title(f'benign: {cls_name}1.png\n{classes[predict]}: {prob:.2%}')
    plt.axis('off')
    plt.imshow(np.array(im))
    # adversarial image
    cnt += 1
    plt.subplot(len(classes), 4, cnt)
    im = Image.open(f'./ifgsm/{path}')
    logit = model(transform(im).unsqueeze(0).to(device))[0]
    predict = logit.argmax(-1).item()
    prob = logit.softmax(-1)[predict].item()
    plt.title(f'adversarial: {cls_name}1.png\n{classes[predict]}: {prob:.2%}')
    plt.axis('off')
    plt.imshow(np.array(im))
plt.tight_layout()
plt.show()

# + [markdown] id="FUmKa02Vmp29"
# ## Report Question
# * Make sure you follow below setup: the source model is "resnet110_cifar10", applying the vanilla fgsm attack on `dog2.png`. You can find the perturbed image in `fgsm/dog2.png`.

# + id="8NW8ntCKY3VY" colab={"base_uri": "https://localhost:8080/", "height": 577} outputId="f0d4f91a-1bf2-4c6f-ad9a-1f9e6a14067b"
# original image
path = f'dog/dog2.png'
im = Image.open(f'./data/{path}')
logit = model(transform(im).unsqueeze(0).to(device))[0]
predict = logit.argmax(-1).item()
prob = logit.softmax(-1)[predict].item()
plt.title(f'benign: dog2.png\n{classes[predict]}: {prob:.2%}')
plt.axis('off')
plt.imshow(np.array(im))
plt.tight_layout()
plt.show()

# adversarial image 
adv_im = Image.open(f'./fgsm/{path}')
logit = model(transform(adv_im).unsqueeze(0).to(device))[0]
predict = logit.argmax(-1).item()
prob = logit.softmax(-1)[predict].item()
plt.title(f'adversarial: dog2.png\n{classes[predict]}: {prob:.2%}')
plt.axis('off')
plt.imshow(np.array(adv_im))
plt.tight_layout()
plt.show()


# + [markdown] id="2AQkofrTnePa"
# ## Passive Defense - JPEG compression
# JPEG compression by imgaug package, compression rate set to 70
#
# Reference: https://imgaug.readthedocs.io/en/latest/source/api_augmenters_arithmetic.html#imgaug.augmenters.arithmetic.JpegCompression

# + id="sKuQaPp2mz7C" colab={"base_uri": "https://localhost:8080/", "height": 297} outputId="98950e90-a3b0-44b2-ca37-a384c70f701c"
import imgaug.augmenters as iaa

# pre-process image
x = transforms.ToTensor()(adv_im)*255
x = x.permute(1, 2, 0).numpy()
x = x.astype(np.uint8)

# TODO: use "imgaug" package to perform JPEG compression (compression rate = 70)
arg = iaa.JpegCompression(compression = 70)
compressed_x =  arg.augment_image(x)


logit = model(transform(compressed_x).unsqueeze(0).to(device))[0]
predict = logit.argmax(-1).item()
prob = logit.softmax(-1)[predict].item()
plt.title(f'JPEG adversarial: dog2.png\n{classes[predict]}: {prob:.2%}')
plt.axis('off')


plt.imshow(compressed_x)
plt.tight_layout()
plt.show()
