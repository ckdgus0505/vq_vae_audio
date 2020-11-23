import os
import subprocess

import math

import matplotlib.pyplot as plt
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import random
from wavenet_vocoder.wavenet import WaveNet
from wavenet_vocoder.wavenet import receptive_field_size
#from vq import VectorQuantizerEMA


# In[2]:


import easydict
args = easydict.EasyDict({
    "length":15872,
    "batch": 1,
    "epochs": 2000,
    "training_data": './2_speaker/vctk_train.txt',
    "test_data": './2_speaker/vctk_test.txt',
#    "training_data": './vctk_train.txt',
#    "test_data": './vctk_test.txt',
#    "out": "result",
#    "resume": False,
    "load": 0,
    "load_mid" : 0,
    "seed": 123456789 })


# In[3]:


device = torch.device("cuda")
torch.cuda.set_device(0)
device

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)


# In[4]:


with open(args.training_data, 'r') as f:
    data = f.read()
file = data.splitlines()
speaker_dic = {}
number_of_speakers = 0
for i in range (0, len(file)):
    if (file[i].split('/')[0] in speaker_dic):
        continue
    else :
        speaker_dic[file[i].split('/')[0]] = number_of_speakers
        number_of_speakers+=1
        


# In[5]:


#TO DO: check that weight gets updated
class VectorQuantizerEMA(nn.Module):
    """We will also implement a slightly modified version  which will use exponential moving averages
    to update the embedding vectors instead of an auxillary loss.
    This has the advantage that the embedding updates are independent of the choice of optimizer 
    for the encoder, decoder and other parts of the architecture.
    For most experiments the EMA version trains faster than the non-EMA version."""
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_()
#        self._embedding.weight.data = torch.Tensor([0])
        #self._embedding.weight.data = torch.Tensor(np.zeros(()))
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon
#    '''
    def forward(self, inputs):
        # convert inputs from BCL -> BLC
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)     #[BL, C]
        if (self._embedding.weight.data == 0).all():
            self._embedding.weight.data = flat_input[-self._num_embeddings:].detach()
        # Calculate distances

        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t())) #[BL, num_embeddings]
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1) #[BL, 1]
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings).to(device)# [BL, num_embeddings]
        encodings.scatter_(1, encoding_indices, 1)
        #print(encodings.shape) [250, 512]
        # Use EMA to update the embedding vectors
#        if self.training:
#            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
#                                     (1 - self._decay) * torch.sum(encodings, 0)
#            #print(self._ema_cluster_size.shape) [512]
#            n = torch.sum(self._ema_cluster_size)
#            self._ema_cluster_size = (
#                (self._ema_cluster_size + self._epsilon)
#                / (n + self._num_embeddings * self._epsilon) * n)
#            
#            dw = torch.matmul(encodings.t(), flat_input)
#            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
#            
#            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Quantize and unflatten
        #encodings.shape = [BL, num_embeddings] , weight.shape=[num_embeddings, C]
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        
        # Loss
        e_latent_loss = torch.mean((quantized.detach() - inputs)**2)
        q_latent_loss = torch.mean((quantized - inputs.detach())**2)
#        print(q_latent_loss.item(), 0.25 * e_latent_loss.item())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
#        loss = self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        # convert quantized from BLC -> BCL
        return loss, quantized.permute(0, 2, 1).contiguous(), perplexity
    '''
    
    def forward(self, inputs):
        # convert inputs from BCL -> BLC
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)     #[BL, C]
        # Calculate distances
        
        distances = torch.norm(flat_input.unsqueeze(1) - self._embedding.weight, dim=2, p=2)
 #       distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
 #                   + torch.sum(self._embedding.weight**2, dim=1)
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1) #[BL, 1]
        print(encoding_indices.squeeze(1))
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings).to(device)# [BL, num_embeddings]
        encodings.scatter_(1, encoding_indices, 1)
        #print(encodings.shape) [250, 512]

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            #print(self._ema_cluster_size.shape) [512]
            n = torch.sum(self._ema_cluster_size)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
          
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
          
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Quantize and unflatten
        #encodings.shape = [BL, num_embeddings] , weight.shape=[num_embeddings, C]
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        # Loss
        e_latent_loss = torch.mean((quantized.detach() - inputs)**2)
        q_latent_loss = torch.mean((quantized - inputs.detach())**2)
#        print(q_latent_loss.item(), 0.25 * e_latent_loss.item())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        # same as torch.exp( entropy loss )
        
        # convert quantized from BLC -> BCL
        return loss, quantized.permute(0, 2, 1).contiguous(), perplexity
    '''


# In[6]:


# embedding_dim=1
# num_embeddings=2
# ema = VectorQuantizerEMA(embedding_dim=embedding_dim,
#                         num_embeddings=num_embeddings,
#                         commitment_cost=0.5,
#                         decay=0.99,
#                         device=device)
  
# ema.eval()
# print("is training", ema.training)
# inputs_np = np.random.randn(20, embedding_dim).astype(np.float32)
# print("inputs", inputs_np)
# inputs = torch.Tensor(inputs_np.reshape(1,embedding_dim,20))

# loss, vq_output, perplexity = ema(inputs)
# print("loss", loss)
# print("output", vq_output)
# # Output shape is correct
# assert vq_output.shape == inputs.shape
    
# #assert ema._embedding.weight.detach().numpy().shape == [embedding_dim, num_embeddings]
# # Check that each input was assigned to the embedding it is closest to.
# embeddings_np = ema._embedding.weight.detach().numpy().T
# distances = ((inputs_np**2).sum(axis=1, keepdims=True) -
#              2 * np.dot(inputs_np, embeddings_np) +
#              (embeddings_np**2).sum(axis=0, keepdims=True))
# closest_index = np.argmax(-distances, axis=1)

# print(closest_index)


# ## Encoder & Decoder Architecture

# In[7]:


class Encoder(nn.Module):
    """Audio encoder
    The vq-vae paper says that the encoder has 6 strided convolutions with stride 2 and window-size 4.
    The number of channels and a nonlinearity is not specified in the paper. 
    I tried using ReLU, it didn't work.
    Now I try using tanh, hoping that this will keep my encoded values within the neighborhood of 0,
    so they do not drift too far away from encoding vectors.
    """
    def __init__(self, encoding_channels, in_channels=256):
        super(Encoder,self).__init__()
        self._num_layers = 2 * len(encoding_channels)
        self._layers = nn.ModuleList()


        self.conv =nn.Sequential(
            nn.Conv1d(256, 512, stride = 2, kernel_size =  4, padding = 1),
            nn.Tanh(),
            nn.Conv1d(512, 512, stride = 2, kernel_size =  4, padding = 1),
            nn.Tanh(),
            nn.Conv1d(512, 512, stride = 2, kernel_size =  4, padding = 1),
            nn.Tanh(),
            nn.Conv1d(512, 512, stride = 2, kernel_size =  4, padding = 1),
            nn.Tanh(),
            nn.Conv1d(512, 512, stride = 2, kernel_size =  4, padding = 1),
            nn.Tanh(),
            nn.Conv1d(512, 512, stride = 2, kernel_size =  4, padding = 1),
            nn.Tanh(),
            nn.Conv1d(512, 512, stride = 2, kernel_size =  4, padding = 1),
            nn.Tanh(),
            nn.Conv1d(512, 80, stride = 2, kernel_size =  4, padding = 1),
            nn.Tanh(),
        )

        for l in self.conv:
            if isinstance(l, nn.Conv1d):
                nn.init.xavier_uniform_(l.weight)

    def forward(self, x):
        return self.conv(x)


# In[8]:


class Model(nn.Module):
    def __init__(self,
                 encoding_channels,
                 num_embeddings, 
                 embedding_dim,
                 commitment_cost, 
                 layers,
                 stacks,
                 kernel_size,
                 decay=0):
        super(Model, self).__init__()       
        self._encoder = Encoder(encoding_channels=encoding_channels)
        #I tried adding batch normalization here, because:
        #the distribution of encoded values needs to be similar to the distribution of embedding vectors
        #otherwise we'll see "posterior collapse": all values will be assigned to the same embedding vector,
        #and stay that way (because vectors which do not get assigned anything do not get updated).
        #Batch normalization is a way to fix that. But it didn't work: model
        #reproduced voice correctly, but the words were completely wrong.
        #self._batch_norm = nn.BatchNorm1d(1)
        if decay > 0.0:
#             self._vq_vae = EMVectorQuantizerEMA(num_embeddings, embedding_dim, 
#                                               commitment_cost, decay, 100)
            self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim, 
                                               commitment_cost, decay)

        else:
            self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                           commitment_cost)
        self._decoder = WaveNet(out_channels=256, #dimension of ohe mu-quantized signal
                                layers=layers, #like in original WaveNet
                                stacks=stacks,
                                residual_channels=512,
                                gate_channels=512,
                                skip_out_channels=256,
                                kernel_size=kernel_size, 
                                dropout=1 - 0.95,
                                cin_channels=embedding_dim, #local conditioning channels - on encoder output
                                gin_channels=number_of_speakers, #global conditioning channels - on speaker_id
                                n_speakers=number_of_speakers,
                                upsample_conditional_features=True,
                                upsample_net="ConvInUpsampleNetwork",
                                upsample_params={"upsample_scales": [4,4,4,4],
                                                 "cin_channels": 80},
                                scalar_input=False,
                                use_speaker_embedding=False,
                                output_distribution='Normal'
                               )
        self.recon_loss = torch.nn.CrossEntropyLoss()
        self.receptive_field = receptive_field_size(total_layers=layers, num_cycles=stacks, kernel_size=kernel_size)
#        self.mean = None
#        self.std = None
    def forward(self, x):
        audio, target, speaker_id = x
        assert len(audio.shape) == 3 # B x C x L 
        assert audio.shape[1] == 256
        z = self._encoder(audio)
        #normalize output - subtract mean, divide by standard deviation
        #without this, perplexity goes to 1 almost instantly
#         if self.mean is None:
#             self.mean = z.mean().detach()
#         if self.std is None:
#              self.std = z.std().detach()
#        z = z - self.mean
#        z = z / self.std
        vq_loss, quantized, perplexity = self._vq_vae(z)
#        assert z.shape == quantized.shape
#        print("audio.shape", audio.shape)
#        print("quantized.shape", quantized.shape)
        x_recon = self._decoder(audio, quantized, speaker_id, softmax=False)
        x_recon = x_recon[:, :, self.receptive_field:-1]
        recon_loss_value = self.recon_loss(x_recon, target[:, 1:])
        loss = recon_loss_value + vq_loss
        
        return loss, recon_loss_value, x_recon, perplexity


# # Train

# In[9]:


num_training_updates = 39818
#vector quantizer parameters:
embedding_dim = 80 #dimension of each vector
encoding_channels = [512,512,512,512,512,512,512,embedding_dim]
num_embeddings = 512 #number of vectors
commitment_cost = 0.25

#wavenet parameters:
kernel_size=3
total_layers=24
num_cycles=4


decay = 0.99
#decay = 0

learning_rate = 1
batch_size=1


# In[10]:


receptive_field = receptive_field_size(total_layers=total_layers, num_cycles=num_cycles, kernel_size=kernel_size)
print(receptive_field)


# ## Load data

# In[11]:


model = Model(num_embeddings=num_embeddings,
              encoding_channels=encoding_channels,
              embedding_dim=embedding_dim, 
              commitment_cost=commitment_cost, 
              layers=total_layers,
              stacks=num_cycles,
              kernel_size=kernel_size,
              decay=decay).to(device)


# In[12]:


optimizer = optim.Adam(model.parameters(), lr=1, amsgrad=False)

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,\
                                              lr_lambda=lambda epoch: 1e-5 \
                                              if epoch == 0 \
                                              else  (optimizer.param_groups[0]['lr'] - (1e-5)/args.epochs) \
                                              if epoch <= args.epochs \
                                              else optimizer.param_groups[0]['lr'])


import librosa


# In[ ]:





# In[15]:


class D_Set(Dataset):
    # VCTK-Corpus Training data set

    def __init__(self, data, num_speakers,
                 receptive_field,
                 segment_length=args.length,
                 chunk_size=1000,
                 classes=256):
        
        self.x_list = self.read_files(data)
        self.classes = 256
        self.segment_length = segment_length
        self.chunk_size = chunk_size
        self.classes = classes
        self.receptive_field = receptive_field
        self.cached_pt = 0
        self.num_speakers = num_speakers

    def read_files(self, filename):
        print("training data from " + args.training_data)
        with open(filename) as file:
            files = file.readlines()
        return [f.strip() for f in files]

    def __getitem__(self, index):
        try:
            audio, sr = librosa.load('./VCTK/wav48/'+self.x_list[index])
        except Exception as e:
            print(e, audiofile)
        if sr != 22050:
            raise ValueError("{} SR of {} not equal to 22050".format(sr, audiofile))
            
        audio = librosa.util.normalize(audio) #divide max(abs(audio))
        audio = self.quantize_data(audio, self.classes)
            
        while audio.shape[0] < self.segment_length:
            index += 1
            audio, speaker_id = librosa.load('./VCTK/wav48/'+self.x_list[index])
            
        max_audio_start = audio.shape[0] - self.segment_length
        audio_start = np.random.randint(0, max_audio_start)
        audio = audio[audio_start:audio_start+self.segment_length]
        
                #divide into input and target
        audio = torch.from_numpy(audio)
        ohe_audio = torch.FloatTensor(self.classes, self.segment_length).zero_()
        ohe_audio.scatter_(0, audio.unsqueeze(0), 1.)
        target = audio[self.receptive_field:]
            
        speaker_index = speaker_dic[self.x_list[index].split('/')[0]]
        speaker_id = torch.from_numpy(np.array(speaker_index)).unsqueeze(0).unsqueeze(0)
        ohe_speaker = torch.FloatTensor(self.num_speakers, 1).zero_()
        ohe_speaker.scatter_(0, speaker_id, 1.)
        
        return ohe_audio, target, ohe_speaker
    
    def __len__(self):
        return len(self.x_list)
    
    def quantize_data(self, data, classes):
        mu_x = self.mu_law_encode(data, classes)
        bins = np.linspace(-1, 1, classes)
        quantized = np.digitize(mu_x, bins) - 1
        return quantized

    def mu_law_encode(self, data, mu):
        mu_x = np.sign(data) * np.log(1 + mu * np.abs(data)) / np.log(mu + 1)
        return mu_x



trainset = D_Set(number_of_speakers, receptive_field=receptive_field)
testset = D_Set(number_of_speakers, receptive_field=receptive_field)


training_loader = DataLoader(args.training_data, dataset = trainset,
                           batch_size=batch_size,
                           shuffle=True, 
                           num_workers=1)


validation_loader = DataLoader(args.training_data, dataset = testset,
                           batch_size=batch_size,
                           shuffle=True, 
                           num_workers=1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[18]:


train_res_recon_error = []
train_res_perplexity = []


# In[ ]:





# In[19]:


def train():
    model.train()
    global train_res_recon_error
    global train_res_perplexity
    train_total_loss = []
    train_recon_error = []
    train_perplexity = []
    # with open("errors", "rb") as file:
    #     train_res_recon_error, train_res_perplexity = pickle.load(file)
# num_epochs = 1
# for epoch in range(num_epochs):
    iterator = iter(training_loader)
#     datas0 = []
#     datas1 = []
#     datas2 = []
    for i, data_train in enumerate(iterator):
        data_train = [data_train[0].to(device),
                     data_train[1].to(device),
                     data_train[2].to(device)
                     ]

#         datas0.append(data_train[0])
#         datas1.append(data_train[1])
#         datas2.append(data_train[2])
#         if (i+1) % batch_size == 0:
#             data = [torch.cat(datas0).to(device),
#                    torch.cat(datas1).to(device),
#                    torch.cat(datas2).to(device)]
        optimizer.zero_grad()
        loss, recon_error, data_recon, perplexity = model(data_train)
        loss.backward()
        optimizer.step()
        train_total_loss.append(loss.item())
        train_recon_error.append(recon_error.item())
        train_perplexity.append(perplexity.item())

        if (i+1) % (10 * batch_size) == 0:
            print('%d iterations' % (i+1))
            print('recon_error: %.3f' % np.mean(train_recon_error[-100:]))
            print('perplexity: %.3f' % np.mean(train_perplexity[-100:]))
            print()
    train_res_recon_error.extend(train_recon_error)
    train_res_perplexity.extend(train_perplexity)
    return np.mean(train_total_loss), np.mean(train_res_recon_error)


# In[20]:


def validation():
    random.seed(args.seed)
    model.eval()
    with torch.no_grad():
        test_total_loss = []
        test_res_recon_error = []
        # with open("errors", "rb") as file:
        #     train_res_recon_error, train_res_perplexity = pickle.load(file)
    # num_epochs = 1
    # for epoch in range(num_epochs):
        iterator = iter(validation_loader)
    #     datas0 = []
    #     datas1 = []
    #     datas2 = []
        for i, data_test in enumerate(iterator):
            data_test = [data_test[0].to(device),
                         data_test[1].to(device),
                         data_test[2].to(device)]
            
            loss, recon_error, data_recon, perplexity = model(data_test)

            test_total_loss.append(loss.item())
            test_res_recon_error.append(recon_error.item())

            if (i+1) % (10 * batch_size) == 0:
                print('%d iterations' % (i+1))
                print('recon_error: %.3f' % np.mean(test_res_recon_error[-100:]))
                print()
    return np.mean(test_total_loss), np.mean(test_res_recon_error)


# In[ ]:





# In[21]:

epochs = args.epochs
training_total_loss_per_epochs = []
training_reconstruction_errors_per_epochs = []
validation_total_loss_per_epochs = []
validation_reconstruction_errors_per_epochs = []

training_mcd_per_epochs = []
validation_mcd_per_epochs = []

lrs = []

if (args.load != 0):
    model.load_state_dict(torch.load("model_epoch"+str(args.load)))
#    optimizer.load_state_dict(torch.load("optim_epoch"+str(args.load)))
    training_total_loss_per_epochs = np.load('training_total_loss_per_epochs.npy').tolist()
    training_reconstruction_errors_per_epochs = np.load('training_reconstruction_errors_per_epochs.npy').tolist()
    validation_total_loss_per_epochs = np.load('validation_total_loss_per_epochs.npy').tolist()
    validation_reconstruction_errors_per_epochs = np.load('validation_reconstruction_errors_per_epochs.npy').tolist()
    lrs = np.load('lrs.npy')
    
    
if (args.load_mid != 0 and args.load == 0):
    model.load_state_dict(torch.load("model_epoch"+str(args.load)))
#    optimizer.load_state_dict(torch.load("optim_epoch"+str(args.load)))


for i in range(1, epochs+1):
    print(str(i)+" epochs ==> training")
    total_loss, reconstruction_loss = train()
    training_total_loss_per_epochs.append(total_loss)
    training_reconstruction_errors_per_epochs.append(reconstruction_loss)
    
    print(str(i)+" epochs ==> validation")
    total_loss, reconstruction_loss = validation()
    validation_total_loss_per_epochs.append(total_loss)
    validation_reconstruction_errors_per_epochs.append(reconstruction_loss)

    
    if (i % 10 == 0):
        torch.save(model.state_dict(), "model_epoch"+str(i+args.load))
#        torch.save(optimizer.state_dict(), "optim_epoch"+str(i+args.load))
        
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        lrs.append(lr)

    np.save('lrs.npy', lrs)
    np.save('training_total_loss_per_epochs', np.array(training_total_loss_per_epochs))
    np.save('training_reconstruction_errors_per_epochs', np.array(training_reconstruction_errors_per_epochs))
    np.save('validation_total_loss_per_epochs', np.array(validation_total_loss_per_epochs))
    np.save('validation_reconstruction_errors_per_epochs', np.array(validation_reconstruction_errors_per_epochs))
    scheduler.step()


# # calculate graph

# In[ ]:




