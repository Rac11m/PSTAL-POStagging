import torch.nn as nn
import tqdm, torch
from model_postag import RNN_postag
from collections import defaultdict
from use_conllulib import CoNLLUReader, Util
from torch.utils.data import TensorDataset, DataLoader
from train_postag import read_corpus
from conllu.serializer import serialize


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

  for sent in words:
    logits = model(torch.LongTensor([sent]))[0] # y_hat
    print([revtagvocab[l.argmax()] for l in logits]) # t_hat
    # print(sentence.serialize())


