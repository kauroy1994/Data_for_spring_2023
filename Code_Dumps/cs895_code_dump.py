#TEST code to implement KG-Mask for KSAT

    
class Tokenizer(object):
    
    @staticmethod    
    def tokenize(text):

        return text.split(' ')

def find_entities(text,kb):

    tokens = Tokenizer.tokenize(text)
    entities = {}

    for token in tokens:
        entities[token] = []

    for token in tokens:
        for kb_item in kb:
            N = len(kb_item)
            if (token in Tokenizer.tokenize(kb_item[0])) or (token in Tokenizer.tokenize(kb_item[2])):
                entities[token] += Tokenizer.tokenize(kb_item[0]) + Tokenizer.tokenize(kb_item[2])
                if len(kb_item) > 3 and (token in Tokenizer.tokenize(kb_item[4])):
                    entities[token] += Tokenizer.tokenize(kb_item[4])
        entities[token] = list(set(entities[token]))
            
    return (entities)
                    

def get_kg_masks(text,kb,ml=10):

    entities = find_entities(text,kb)
    tokens = Tokenizer.tokenize(text)
    
    N = len(tokens)
    mask = {}
    for token in entities:
        mask[token] = [0.0]*ml
        
    for token in entities:
        for related_tokens in entities[token]:
            for i in range(N):
                if tokens[i] == related_tokens:
                    mask[token][i] = 1.0

    return mask
    


#test sentence
test = 'greenhouse effect is a type of atmospheric phenomenon'
#test = 'I dont like the greenhouse effect'

#test kb
kb = [('greenhouse effect','is a type of','atmospheric phenomenon'),
      ('greenhouse effect','is a', 'effect','is a type of','atmospheric phenomenon')]


KG_masks = get_kg_masks(test,kb)

#======================================================================================
import torch
import torch.nn as nn

#KSAT class
class KSAT(nn.Module):

    def __init__(self,
                 data = None,
                 kb = None,
                 ml = 10,
                 model='shallow'):

        super().__init__()

        self.data = data
        self.kb = kb
        self.model = model
        self.ml = ml

    def find_entities(self,
                      x):

        kb = self.kb
        text = x

        tokens = Tokenizer.tokenize(text)
        entities = {}

        for token in tokens:
            entities[token] = []

        for token in tokens:
            for kb_item in kb:
                N = len(kb_item)
                if (token in Tokenizer.tokenize(kb_item[0])) or (token in Tokenizer.tokenize(kb_item[2])):
                    entities[token] += Tokenizer.tokenize(kb_item[0]) + Tokenizer.tokenize(kb_item[2])
                    if len(kb_item) > 3 and (token in Tokenizer.tokenize(kb_item[4])):
                        entities[token] += Tokenizer.tokenize(kb_item[4])
            entities[token] = list(set(entities[token]))
            
        return (entities)

    def get_kg_masks(self,
                     x):

        text = x
        kb = self.kb
        ml = self.ml

        entities = find_entities(text,kb)
        tokens = Tokenizer.tokenize(text)
    
        N = len(tokens)
        mask = {}
        for token in entities:
            mask[token] = [0.0]*ml
        
        for token in entities:
            for related_tokens in entities[token]:
                for i in range(N):
                    if tokens[i] == related_tokens:
                        mask[token][i] = 1.0

        masks = []
        masks += [mask[token] for token in tokens]
        masks += [[0.0]*ml for i in range(ml-N)]

        return torch.tensor(masks)

    def GCN(self,
            x):

        mask = self.get_kg_masks(x) 

    def embed(self,
              x):

        #tokenize and construct one-hot encoding
        
        tokens = Tokenizer.tokenize(x)
        N = len(tokens)
        encodings = [[0.0]*self.ml for i in range(N)]

        for i in range(N):
            encodings[i][i] = 1.0

        kg_mask = self.get_kg_masks(x)
        encodings = torch.tensor(encodings)

        for i in range(2):
            encodings = encodings @ kg_mask

        return encodings


    def forward(self,
                x):

        
        input_repr = self.embed(x)
        print (input_repr)

#Unit test
m = KSAT(kb=kb)
m(test)

# -*- coding: utf-8 -*-
"""CS895_code_dump.ipynb

#Download simple transformers library
"""

!pip install simpletransformers

"""#Test Cuda availability and assign device"""

#import pytorch
import torch

#assign device as 'cpu' or 'cuda'
device = ('cuda' if torch.cuda.is_available() else 'cpu')

"""#Get data and concepts from google drive"""

from google.colab import drive

#mount google drive
drive.mount("/content/gdrive")

#path to data and concepts file
path = "gdrive/MyDrive/AGNNs/"

#import pickle to read in binary files
import pickle
data_binary = open(path+"data.pkl",'rb')
concepts_binary = open(path+"concept_paths.pkl",'rb')

#read in data and concept paths
data = pickle.load(data_binary)
concept_paths = pickle.load(concepts_binary)

"""#Data and concepts preprocessing """

#get all concepts
concepts = list(concept_paths.keys())[:]

#format paths as strings
string_paths = []
for concept in concepts:
  concept_path = concept_paths[concept]

  #remove manually observed noise issues and store path
  if concept_path and concept_path[0] and len(concept_path) > 0:
    if not [x for x in concept_path if '[\'\'' in x]:
      for path in concept_path:
        if path.replace(',','').replace('\'','')[1:-1]:
          string_paths.append(path.replace(',','').replace('\'','')[1:-1])

#make concept_data
concept_data = [(path,float(1.0)) for path in string_paths]

"""#Moment-based training function definitions
#1. Supervised learning (SVL)
#2. Weighted supervised learning (WSVL)
#3. Vanilla policy gradients (VPG)
#4. Weighted policy gradients (WPG)
#5. Knowledge infused policy gradients (KIPG)
"""

#import embedding mode
from simpletransformers.language_representation import RepresentationModel

#import pytorch's neural net module
import torch.nn as nn

#import training helper functions
from tqdm import tqdm
from random import sample,choice,shuffle
from torch.optim import SGD
from sklearn.metrics import accuracy_score

#Moment-BERT (MERT) class
class MERT(nn.Module):

  @staticmethod
  def normalize(X):

    mean = X.mean()
    sigma = X.std()

    Z = torch.nan_to_num((X-mean)/sigma)
    return Z

  @staticmethod
  def get_batch(knowledge=True):

    '''returns data batch
       of size 32
    '''

    #set knowledge = False if dont want knowledge
    if not knowledge:
      batch = sample(data,32)
      shuffle(batch)
      return (batch)

    #sample 16 size concept paths batch
    concept_batch = sample(concept_data,16)

    #append weights to the batch of concept paths
    concet_batch = [(x[0],x[1],16.0/272.0) for x in concept_batch]

    #sample 16 size data batch
    data_batch = sample(data,16)

    #append weights to the data batch
    data_batch = [(x[0],x[1],1.0/272.0) for x in data_batch]

    #aggregate the data and concept paths, shuffle and return
    batch = concept_batch + data_batch
    shuffle(batch)
    return batch

  def __init__(self,
               moment_order = 50):
    
    #initialize constructor for nn module
    super().__init__()

    #check if device = cuda
    cuda  = (device if device == 'cuda' else False)

    #set moment order
    self.M = moment_order

    #initialize BERT embedding model
    self.model = RepresentationModel(model_type='bert',
                                     model_name='bert-base-uncased',
                                     use_cuda=cuda)
    
    #get token embedding size from test encoding
    t_en = self.model.encode_sentences(['test'],
                                       combine_strategy='mean')[0]
    self.e_size = len(t_en)
    
    #sample 10 train and eval batches
    self.train_batches = [MERT.get_batch() for i in range(10)]
    self.eval_batches = [MERT.get_batch() for i in range(10)]

    #set hidden layer weights
    h_size = self.M*self.e_size
    self.h_weights = nn.Parameter(torch.rand(h_size,h_size),
                                  requires_grad=True)
    
    #set layer norm weights
    self.lay_weights = nn.Parameter(torch.rand(h_size,1),
                                    requires_grad=True)

    #set layer norm biases
    self.lay_biases = nn.Parameter(torch.rand(h_size,1),
                                   requires_grad=True)
    
    #set activation functions
    self.relu, self.sigmoid = nn.LeakyReLU(), nn.Sigmoid()

  def embed(self,
            x):

    #get input embeddings
    e_x = self.model.encode_sentences([x])[0]

    #convert to torch tensor and add 1.0 to prevent underflow
    e_x = torch.add(torch.tensor(e_x),1.0)
    M_x = [torch.pow(e_x,m).mean(dim=0) for m in range(self.M)]
    M_x = torch.row_stack(M_x)
    M_x = torch.reshape(M_x,(self.M*self.e_size,1))
    return M_x

  def forward(self,
              x):
    
    #get moments
    M_x = self.embed(x)

    #pass through non-linear hidden layer
    h_x = self.relu(self.h_weights @ M_x)

    #layer normalize
    h_x = MERT.normalize(h_x)
    h_x = self.lay_weights * h_x
    h_x = self.lay_biases * h_x

    #dot product and squash
    z_x = self.sigmoid(h_x.t() @ h_x)

    #return squashed output
    return z_x

"""#Unit tests"""

mert = MERT(moment_order=10)
print (mert('test'))

"""#Training function definitions
#1. Supervised learning (SVL)
#2. Weighted supervised learning (WSVL)
#3. Vanilla policy gradients (VPG)
#4. Weighted policy gradients (WPG)
#5. Knowledge infused policy gradients (KIPG)
"""

#import embedding model
from simpletransformers.language_representation import RepresentationModel

#import pytorch's neural net module
import torch.nn as nn

#import training helper functions
from tqdm import tqdm
from random import sample,choice,shuffle
from torch.optim import SGD
from sklearn.metrics import accuracy_score

#BERT baseline class
class BERT(nn.Module):

  @staticmethod
  def normalize(X):
    
    mean = X.mean()
    sigma = X.std()
    
    Z = torch.nan_to_num((X-mean)/sigma)
    return Z

  @staticmethod
  def get_batch(knowledge=True):

    '''returns data batch
       of size 32
    '''

    #set knowledge = False if dont want knowledge
    if not knowledge:
      batch = sample(data,32)
      shuffle(batch)
      return (batch)

    #sample 16 size concept paths batch
    concept_batch = sample(concept_data,16)

    #append weights to the batch of concept paths
    concept_batch = [(x[0],x[1],16.0/272.0) for x in concept_batch]

    #sample 16 size data batch
    data_batch = sample(data,16)

    #append weights to the data batch
    data_batch = [(x[0],x[1],1.0/272.0) for x in data_batch]

    #aggregate the data and concept paths, shuffle, and return
    batch = concept_batch + data_batch
    shuffle(batch)
    return (batch)

  def __init__(self):

    #initialize constructor for nn module
    super().__init__()
    
    #check if device = cuda
    cuda = (device if device == 'cuda' else False)
    
    #initialize BERT embedding model
    self.model = RepresentationModel(model_type='bert',
                                     model_name='bert-base-uncased',
                                     use_cuda=cuda)

    #sample 10 train and eval batches
    self.train_batches = [BERT.get_batch() for i in range(10)]
    self.eval_batches = [BERT.get_batch() for i in range(10)]

    #set max_token_length
    self.ml = 512

    def embed(x):

      #get input embeddings
      e_x = self.model.encode_sentences([x])[0]

      #convert to torch tensor and return
      e_x = torch.tensor(e_x)

      #pad token vectors and row stack
      pads = [torch.zeros(e_x.shape[1]) for i in range(self.ml-e_x.shape[0])]
      e_x = torch.row_stack([e_x]+pads)

      return e_x

    #set embedding function
    self.embed = embed

    #encode dummy token to get size
    self.h_size = self.embed('dummy').size()[1]

    #set hidden layer weights
    self.hid_weights = nn.Parameter(torch.rand(self.h_size,self.h_size),
                                    requires_grad=True)
    
    #set layer norm weights
    self.lay_weights = nn.Parameter(torch.rand(self.ml,self.h_size),
                                    requires_grad=True)
    
    #set layer norm biases
    self.lay_biases = nn.Parameter(torch.rand(self.ml,self.h_size),
                                    requires_grad=True)

    #set linear layer weights
    self.lin_weights = nn.Parameter(torch.rand(self.ml,self.h_size),
                                    requires_grad=True)
    
    #set linear layer bias
    self.lin_bias = nn.Parameter(torch.rand(1),
                                 requires_grad=True)
    
    #set layer norm function
    self.normalize = BERT.normalize

    #set activation functions
    self.relu, self.sigmoid = nn.LeakyReLU(), nn.Sigmoid()

  def forward(self,
              x):
    
    #get embeddings,
    e_x = self.embed(x)

    #pass through non-linear hidden layer
    h_x = self.relu(e_x @ self.hid_weights)

    #layer normalize
    h_x = self.normalize(h_x)
    h_x = self.lay_weights * h_x
    h_x = self.lay_biases * h_x

    #pass through linear layer and squash
    z_x = torch.sum(h_x * self.lin_weights)
    z_x = z_x * self.lin_bias
    z_x = self.sigmoid(z_x)

    #return squashed output
    return z_x

  @staticmethod
  def eval(model,
           infer_func = None,
           KIPG = False):

    #initialize avg accuracy
    accs = 0.0

    #if model is a KIPG model
    if KIPG:

      #get the KIPG models
      models = KIPG 
      
      #calculate evaluation accuracy across batches and sum
      for batch in model.eval_batches:
        y_hat = [float(infer_func(x[0],models) >= 0.5) for x in batch]
        y = [float(x[1] >= 0.5) for x in batch]
        accs += accuracy_score(y,y_hat)

      #calculate average accuracy
      accs /= len(model.eval_batches)

      #print evaluation accuracy and return
      print ("eval acc", accs)
      return (accs)

    #calculate evaluation accuracy across batches and sum
    for batch in model.eval_batches:
      y_hat = [float(model(x[0]) >= 0.5) for x in batch]
      y = [float(x[1] >= 0.5) for x in batch]
      accs += accuracy_score(y,y_hat)

    #calculate average accuracy
    accs /= len(model.eval_batches) 

    #print evaluation accuracy and return
    print ("eval acc", accs)
    return (accs)

  @staticmethod
  def WPG(hyperparams = False,
          lrs = [0.001],
          epochs = [10],
          eps = 1e-10):
    '''method to train the BERT model
       using weighted policy gradients
       (WPG)
    '''

    #set loss
    loss = nn.BCELoss()

    #read in hyperparams if provided
    if hyperparams:

      #read in learning rates
      if 'lrs' in hyperparams:
        lrs = hyperparams['lrs']

      #read in epochs
      if 'epochs' in hyperparams:
        epochs = hyperparams['epochs']

    #initialize policy checkpoints
    BERT.policy_checkpoints = []
    policy_checkpoints = BERT.policy_checkpoints


    for i in tqdm(lrs):
      print ('='*40)
      print ("learning rate", i)
      print ('='*40)

      for j in tqdm(epochs):

        #get BERT model
        bert = BERT()

        #get optimizer
        optimizer = SGD(bert.parameters(),lr=i)

        #training loop
        for k in range(j):

          #sample an episode from training batches
          episode = choice(bert.train_batches)

          #get episode length
          N = len(episode)

          #negative Q-values = negative of one-step reward over episode observations
          outs = [bert(obs[0]) for obs in episode]
          ys = [torch.unsqueeze(torch.tensor(obs[1]),0) for obs in episode]
          neg_Qs = torch.stack([loss(r[0],r[1]) for r in zip(outs,ys)])

          #get log probabilities
          log_probs = torch.stack([torch.log(out+eps) for out in outs])

          #uncomment below to normalize neg_Qs if needed
          #neg_Qs = bert.normalize(neg_Qs)

          #obtain episode observation weights
          ws = torch.tensor([obs[2] for obs in episode])

          #compute weighted negative avg cumulative episode reward, avg for numerical stability
          neg_episode_Q = -(ws * log_probs * neg_Qs).mean()

          #print episode_Q
          print ("avg episode reward", -1 * neg_episode_Q)
          
          #perform gradient ascent step
          neg_episode_Q.backward()
          optimizer.step()
          optimizer.zero_grad()

          #store model in policy checkpoints list
          policy_checkpoints.append((bert,BERT.eval(bert),k,i))

    #search and return best policy
    best_metrics = max([checkpoint[1] for checkpoint in policy_checkpoints])
    best_checkpoint = (checkpoint for checkpoint in policy_checkpoints if checkpoint[1]==best_metrics)
    return best_checkpoint

  @staticmethod
  def VPG(hyperparams = False,
          lrs = [0.001],
          epochs = [10],
          eps = 1e-10):
    '''method to train the BERT model
       using vanilla policy gradients
       (VPG)
    '''

    #set loss
    loss = nn.BCELoss()

    #read in hyperparams if provided
    if hyperparams:

      #read in learning rates
      if 'lrs' in hyperparams:
        lrs = hyperparams['lrs']

      #read in epochs
      if 'epochs' in hyperparams:
        epochs = hyperparams['epochs']

    #initialize policy checkpoints
    BERT.policy_checkpoints = []
    policy_checkpoints = BERT.policy_checkpoints


    for i in tqdm(lrs):
      print ('='*40)
      print ("learning rate", i)
      print ('='*40)

      for j in tqdm(epochs):

        #get BERT model
        bert = BERT()

        #get optimizer
        optimizer = SGD(bert.parameters(),lr=i)

        #training loop
        for k in range(j):

          #sample an episode from training batches
          episode = choice(bert.train_batches)

          #get episode length
          N = len(episode)

          #negative Q-values = negative of one-step reward over episode observations
          outs = [bert(obs[0]) for obs in episode]
          ys = [torch.unsqueeze(torch.tensor(obs[1]),0) for obs in episode]
          neg_Qs = torch.stack([loss(r[0],r[1]) for r in zip(outs,ys)])

          #get log probabilities
          log_probs = torch.stack([torch.log(out+eps) for out in outs])

          #uncomment below to normalize neg_Qs if needed
          #neg_Qs = bert.normalize(neg_Qs)

          #compute negative avg cumulative episode reward, avg for numerical stability
          neg_episode_Q = -(log_probs * neg_Qs).mean()

          #print episode_Q
          print ("avg episode reward", -1 * neg_episode_Q)
          
          #perform gradient ascent step
          neg_episode_Q.backward()
          optimizer.step()
          optimizer.zero_grad()

          #store model in policy checkpoints list
          policy_checkpoints.append((bert,BERT.eval(bert),k,i))

    #search and return best policy
    best_metrics = max([checkpoint[1] for checkpoint in policy_checkpoints])
    best_checkpoint = (checkpoint for checkpoint in policy_checkpoints if checkpoint[1]==best_metrics)
    return best_checkpoint

  @staticmethod
  def WSVL(hyperparams = False,
           lrs = [0.001],
           epochs = [10]):
    '''method to train the BERT model
       using weighted supervised learning
       WSVL
    '''

    #set loss
    loss = nn.BCELoss()
    
    #read in hyperparams if provided
    if hyperparams:

      #read in learning rates
      if 'lrs' in hyperparams:
        lrs = hyperparams['lrs']

      #read in epochs
      if 'epochs' in hyperparams:
        epochs = hyperparams['epochs']

    #initialize model checkpoints
    BERT.model_checkpoints = []
    model_checkpoints = BERT.model_checkpoints

    for i in tqdm(lrs):
      print ('='*40)
      print ("learning rate", i)
      print ('='*40)

      for j in tqdm(epochs):

        #get BERT model
        bert = BERT()

        #get optimizer
        optimizer = SGD(bert.parameters(),lr=i)

        #training loop
        for k in range(j):

          #sample training batch
          batch = choice(bert.train_batches)

          #get batch size
          N = len(batch)

          #initialize batch loss
          batch_loss = 0.0

          for item in batch:

            #get example
            x,y = item[0],item[1]

            #get example weight
            w = item[1]

            #convert to torch tensor
            y = torch.unsqueeze(torch.tensor(y),0)
            
            #compute forward pass and example loss
            out = bert(x)
            out_loss = loss(out,y)

            #weight and add to batch_loss
            batch_loss += w * out_loss

          #average batch loss for numerical stability
          batch_loss /= N
          
          #print loss
          print ("batch loss", batch_loss)

          #compute gradients
          batch_loss.backward()

          #gradient descent step
          optimizer.step()

          #comment out below to accumulate gradients
          optimizer.zero_grad()

          #store model in checkpoints list
          model_checkpoints.append((bert,BERT.eval(bert),k,i))

    #search and return best performing model
    best_metrics = max([checkpoint[1] for checkpoint in model_checkpoints])
    best_checkpoint = (checkpoint for checkpoint in model_checkpoints if checkpoint[1]==best_metrics)
    return best_checkpoint

  @staticmethod
  def SVL(hyperparams = False,
          lrs = [0.001],
          epochs = [10]):
    '''method to train the BERT model
       using supervised learning
       SVL
    '''

    #set loss
    loss = nn.BCELoss()
    
    #read in hyperparams if provided
    if hyperparams:

      #read in learning rates
      if 'lrs' in hyperparams:
        lrs = hyperparams['lrs']

      #read in epochs
      if 'epochs' in hyperparams:
        epochs = hyperparams['epochs']

    #initialize model checkpoints
    BERT.model_checkpoints = []
    model_checkpoints = BERT.model_checkpoints

    for i in tqdm(lrs):
      print ('='*40)
      print ("learning rate", i)
      print ('='*40)

      for j in tqdm(epochs):

        #get BERT model
        bert = BERT()

        #get optimizer
        optimizer = SGD(bert.parameters(),lr=i)

        #training loop
        for k in range(j):

          #sample training batch
          batch = choice(bert.train_batches)

          #get batch size
          N = len(batch)

          #initialize batch loss
          batch_loss = 0.0

          for item in batch:

            #get example
            x,y = item[0],item[1]

            #convert to torch tensor
            y = torch.unsqueeze(torch.tensor(y),0)
            
            #compute forward pass and example loss
            out = bert(x)
            out_loss = loss(out,y)

            #add to batch_loss
            batch_loss += out_loss
          
          #average batch loss for numerical stability
          batch_loss /= N

          #print loss
          print ("batch loss", batch_loss)

          #compute gradients
          batch_loss.backward()

          #gradient descent step
          optimizer.step()

          #comment out below to accumulate gradients
          optimizer.zero_grad()

          #store model in checkpoints list
          model_checkpoints.append((bert,BERT.eval(bert),k,i))

    #search and return best performing model
    best_metrics = max([checkpoint[1] for checkpoint in model_checkpoints])
    best_checkpoint = (checkpoint for checkpoint in model_checkpoints if checkpoint[1]==best_metrics)
    return best_checkpoint

  @staticmethod
  def KIPG(hyperparams=False,
           lrs = [1],
           steps = [10],
           eps = 1e-10):
    '''method to train the BERT model
       using knowledge infused policy gradients
       (KIPG)
    '''

    #set loss
    loss = nn.BCELoss()

    #read in hyperparams if provided
    if hyperparams:

      #read in learning rates
      if 'lrs' in hyperparams:
        lrs = hyperparams['lrs']

      #read in gradient steps
      if 'steps' in hyperparams:
        steps = hyperparams['steps']

    #initialize policy checkpoints
    BERT.policy_checkpoints = []
    policy_checkpoints = BERT.policy_checkpoints

    #define inference over gradient models
    def infer(x,
              lr,
              models = []):
      
      y = torch.zeros(1,1)

      if not models:
        return bert.sigmoid(y)

      for model in models:

        e_x = bert.embed(x)
        h_x = bert.relu(e_x @ model['h_weights'])
        h_x = torch.reshape(h_x,(bert.ml * bert.h_size,1))
        h_x_t = h_x.t()

        y += lr * (h_x_t @ model['lstsq_weights'])

      return bert.sigmoid(y)

    #define gradient computation function
    def compute_grad(episode,
                     lr,
                     models = []):
      
      #randomly initialize hidden layer weights
      h_weights = torch.rand(bert.h_size,bert.h_size)

      #compute data points for lin-reg
      X,Y = [],[]

      #initialize episode Q
      episode_Q = []

      #get max weight
      max_w = max([obs[2] for obs in episode])

      for obs in episode:
        e_x = bert.embed(obs[0])
        h_x = bert.relu(e_x @ h_weights)
        h_x = torch.reshape(h_x,(bert.ml * bert.h_size,1))
        X.append(h_x.t())

        #weighted observation functional gradient
        out = infer(obs[0],lr,models)
        y = torch.tensor([[obs[1]]])
        neg_Q = -loss(out,y)
        episode_Q.append(neg_Q)

        I = int(obs[2] == max_w)

        obs_grad = -((obs[1] - infer(obs[0],lr,models)+I)*neg_Q)
        Y.append(obs_grad)

      #print episode_Q
      episode_Q = torch.stack(episode_Q)
      print ("avg episode reward",episode_Q.mean())

      X = torch.row_stack(X)
      Y = torch.row_stack(Y)

      #compute weights using least squares projection
      lstsq_weights = torch.linalg.lstsq(X,Y).solution

      #return functional gradient model
      grad_model = [{'h_weights': h_weights,
                     'lstsq_weights': lstsq_weights}]
      return models+grad_model

    for i in tqdm(lrs):

      print ('='*40)
      print ('learning rate', i)
      print ('='*40)

      for j in tqdm(steps):

        #get BERT model
        bert = BERT()

        #initialize gradient models
        grad_models = []

        #training loop
        for k in range(j):

          #sample an episode from training batches
          episode = choice(bert.train_batches)

          #get episode length
          N = len(episode)

          #compute functional gradients
          grad_models = compute_grad(episode,
                                     i,
                                     grad_models)

          #store model in policy checkpoints list
          policy_checkpoints.append((grad_models,BERT.eval(bert,infer,grad_models),k,i))

    #search and return best policy
    best_metrics = max([checkpoint[1] for checkpoint in policy_checkpoints])
    best_checkpoint = (checkpoint for checkpoint in policy_checkpoints if checkpoint[1]==best_metrics)
    return best_checkpoint

"""#Training BERT using knowledge infused policy gradients (KIPG)"""

model = BERT.KIPG()
#model = BERT.KIPG(hyperparams={'steps':[10,20,50]})

"""#Training BERT using supervised learning (SVL)


"""

model = BERT.SVL()
#model = BERT.SVL(hyperparams={'lrs':[0.001,0.005,0.01,0.05,0.1,0.5,1.0]})

"""#Training BERT using weighted supervised learning (WSVL)"""

model = BERT.WSVL()
#model = BERT.WSVL(hyperparams={'lrs':[0.001,0.005,0.01,0.05,0.1,0.5,1.0]})

"""#Training BERT using vanilla policy gradients (VPG)"""

model = BERT.VPG()
#model = BERT.VPG(hyperparams={'lrs':[0.001,0.005,0.01,0.05,0.1,0.5,1.0]})

"""#Training BERT using weighted policy gradients (WPG)"""

model = BERT.WPG()
#model = BERT.VPG(hyperparams={'lrs':[0.001,0.005,0.01,0.05,0.1,0.5,1.0]})
