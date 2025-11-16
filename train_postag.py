import torch.nn as nn
import tqdm, sys, torch
from traitlets import default
from pandas import read_clipboard
import model_postag, use_conllulib
from model_postag import RNN_postag
from collections import defaultdict
from use_conllulib import CoNLLUReader
from torch.utils.data import TensorDataset, DataLoader


def pad_tensor(X, max_len):
  res = torch.full((len(X), max_len), 0)
  for (i, row) in enumerate(X) :
    x_len = min(max_len, len(X[i]))
    res[i,:x_len] = torch.LongTensor(X[i][:x_len])
  return res

def fit(model, epochs, train_loader, dev_loader):
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters()) 
  for epoch in range(epochs):
    model.train()
    total_loss = 0
    for (X, y) in tqdm.tqdm(train_loader) :      
      optimizer.zero_grad()
      y_hat = model(X)    
      loss = criterion(y_hat, y)
      loss.backward()
      optimizer.step()
      total_loss += loss.item()  
    print(f"{epoch}/{epochs} ")
    print("train_loss = {:.4f}".format(total_loss / len(train_loader.dataset)))
    print("dev_loss = {:.4f} dev_acc = {:.4f}".format(*perf(model, dev_loader, criterion)))

def perf(model, dev_loader, criterion):
  model.eval()
  total_loss = correct = 0
  for (X, y) in dev_loader:
    with torch.no_grad():
      y_hat = model(X) 
      total_loss += criterion(y_hat, y)
      y_pred = torch.max(y_hat, dim=1)[1] # argmax
      mask = (y_pred != model.PAD_ID)
      correct += torch.sum((y_pred.data == y) * mask)
  total = len(dev_loader.dataset)
  return total_loss / total, correct / total


def read_corpus(filename, wordvocab, tagvocab, max_len, batch_size, train_mode=True, batch_mode=True):
    if train_mode:
      wordvocab = defaultdict(lambda: len(wordvocab))
      wordvocab["<PAD>"]
      wordvocab["<UNK>"]

      tagvocab = defaultdict(lambda: len(tagvocab))
    
    words, tags = [], []

    infile = open(filename, encoding='UTF-8')
    conllu_reader = CoNLLUReader(infile=infile)

    for sent in conllu_reader.readConllu():
      
      tag = sent[0]['upos']
      if train_mode:
        tags.append(tagvocab[tag])
      else:
        tags.append(tagvocab.get(tag, tagvocab["<PAD>"]))
      
      forms = [tok["form"] for tok in sent]

      if train_mode:
        words.append([wordvocab[w] for w in forms])
      else:
        words.append([wordvocab.get(w, wordvocab['<UNK>']) for w in forms])
      
    infile.close()

    if batch_mode:
      dataset = TensorDataset(pad_tensor(words, max_len), torch.LongTensor(tags))
      dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train_mode)
      return dataloader, wordvocab, tagvocab
    else:
      return words, tags, wordvocab, tagvocab

if __name__ == "__main__" : 
  train_loader, wordvocab, tagvocab = read_corpus(filename="../pstal-etu/sequoia/sequoia-ud.parseme.frsemcor.simple.train", wordvocab=None, tagvocab=None, max_len=40, batch_size=32, train_mode=True, batch_mode=True)
  num_embeddings= len(wordvocab) 
  output_size= len(tagvocab) 
  hidden_size=200
  embedding_dim=250
  hp = {
    "model_type": "GRU", 
    "embedding_dim": embedding_dim, 
    "hidden_size": hidden_size}
  
  dev_loader, _, _ = read_corpus(filename="../pstal-etu/sequoia/sequoia-ud.parseme.frsemcor.simple.dev", wordvocab=wordvocab, tagvocab=tagvocab, max_len=40, batch_size=32, train_mode=False, batch_mode=True)
  
  model = RNN_postag(hidden_size=hidden_size, output_size=output_size, num_embeddings=num_embeddings, embedding_dim=embedding_dim, PAD_ID=0)
  
  fit(model=model, epochs=15, train_loader=train_loader, dev_loader=dev_loader)
  
  torch.save({"wordvocab": dict(wordvocab), 
              "tagvocab": dict(tagvocab), 
              "model_params": model.state_dict(),
              "hyperparams": hp}, "model.pt")
