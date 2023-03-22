import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from random import sample, shuffle

class Tokenizer(object):

    def __init__(self,
                 tokens = None):

        self.tokens = tokens
        self.n_tokens = len(tokens)

    def encode(self,
               text):

        chars = list(text)
        return ([self.tokens.index(c) for c in chars])

    def decode(self,
               text_encoding):

        return ''.join([self.tokens[encoding] for encoding in text_encoding])

class Dataloader(object):

    def __init__(self,
                 tokenizer = None,
                 text = None):

        self.context_size = len(list(text))
        X, Y  = [],[]

        print ('Loading the data ...')
        for t in tqdm(range(self.context_size-1)):

            x, y = tokenizer.encode(text[:t+1]), tokenizer.encode(text[t+1])
            X += [x]; Y += [y[0]]

        self.data = list([list(item) for item in zip(X,Y)])

    def get_batch(self,
                  n = None):

        shuffle(self.data)
        return self.data
        

class ffn_complete(nn.Module):

    def __init__(self,
                 n_tokens = None,
                 emb_size = None,
                 context_size = None,
                 n_layers = 2,
                 h_size = 100):

        super().__init__()

        self.n_tokens = n_tokens
        self.emb_size = emb_size
        self.context_size = context_size
        self.n_layers = n_layers
        self.h_size = h_size

        self.embeddings = nn.Embedding(self.n_tokens, self.emb_size)
        self.pos_embeddings = nn.Embedding(self.context_size, self.emb_size)

        self.fc1 = nn.Linear(self.emb_size, self.h_size, bias = False)
        self.fc2 = nn.Linear(self.h_size, self.h_size, bias = False)
        self.head = nn.Linear(self.h_size, self.n_tokens)

    def forward(self,
                token_encodings):

        n_tokens = len(token_encodings)
        token_encodings = torch.tensor(token_encodings)
        token_embeddings = self.embeddings(token_encodings)
        pos_embeddings = self.pos_embeddings(torch.arange(n_tokens))
        token_embeddings += pos_embeddings
        token_embeddings = F.leaky_relu(self.fc1(token_embeddings))
        token_embeddings = F.leaky_relu(self.fc2(token_embeddings))
        token_embeddings = self.head(token_embeddings)

        logits = token_embeddings[-1]
        return logits

    def generate(self,
                 x):

        for i in range(10000):

            x = x[:self.context_size]
            logits = self(x)
            logits = F.softmax(logits,dim=-1)
            next_id = torch.multinomial(logits,num_samples = 1)
            x = x + next_id.tolist()

        return tokenizer.decode(x)

    def train(self,
              data_loader):

        optimizer = torch.optim.AdamW(self.parameters())

        print ('training the autocompletion neural net ...')
        for i in tqdm(range(1000)):

            batch = data_loader.get_batch()
            n_batch = len(batch)
            loss = F.cross_entropy

            batch_loss = 0.0
            for item in batch:

                x, y = item[0], item[1]
                #logits = self(x)
                logits = torch.unsqueeze(self(x),0)
                #targets = [0.0]*self.n_tokens; targets[y] = 1.0
                targets = [y]
                targets = torch.tensor(targets)
                batch_loss += loss(logits, targets)

            batch_loss /= n_batch
            #print (batch.loss.item())
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
