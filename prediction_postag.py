import torch.nn as nn
import tqdm, torch
from model_postag import RNN_postag
from collections import defaultdict
from use_conllulib import CoNLLUReader, Util
from torch.utils.data import TensorDataset, DataLoader
from train_postag import read_corpus
from conllu.serializer import serialize
from conllu.models import TokenList


if __name__ == "__main__" :  
  
  load_dict = torch.load("model.pt", weights_only=False)
  wordvocab = load_dict["wordvocab"]
  num_embeddings= len(wordvocab) 
  tagvocab = load_dict["tagvocab"]
  output_size= len(tagvocab) 
  hp = load_dict["hyperparams"]

  model = RNN_postag(embedding_dim=hp["embedding_dim"], hidden_size=hp["hidden_size"], num_embeddings=num_embeddings, output_size=output_size)
  model.load_state_dict(load_dict["model_params"])              
  
  words, _, _, _ = read_corpus(filename="../pstal-etu/sequoia/sequoia-ud.parseme.frsemcor.simple.dev", wordvocab=wordvocab, tagvocab=tagvocab, max_len=40, batch_size=32, train_mode=False, batch_mode=False)
  revtagvocab = Util.rev_vocab(tagvocab)
  revwordvocab = Util.rev_vocab(wordvocab)

  revwords = []
  for sent in words:
    revwords.append([revwordvocab[s] for s in sent])

  sentences = []

  for sent in words:
    logits = model(torch.LongTensor([sent]))[0] # y_hat
    forms = [revwordvocab[w] for w in sent] 
    upos = [revtagvocab[l.argmax()] for l in logits]
    
    conllu_format = [{"id": index, "form": w, "upos": t} for index, (w, t) in enumerate(zip(forms, upos), start=1)]
    sentences.append(TokenList(conllu_format)) 
  
  with open('sequoia-ud.parseme.frsemcor.simple.pred', 'w', encoding="utf-8") as f:
    f.writelines([sentence.serialize() + "\n" for sentence in sentences])
  
