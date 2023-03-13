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

        
