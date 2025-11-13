import tqdm
import torch
from traitlets import default
import model_postag, use_conllulib
from use_conllulib import CoNLLUReader
import torch.nn as nn
from collections import defaultdict
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


def read_corpus(filename):
    infile = open(filename, encoding='UTF-8')
    conllu_reader = CoNLLUReader(infile=infile) 

    col_name_dict = {
        "form": ["<PAD>", "<UNK>"],  # tokens (words)
        "upos": ["<PAD>"]            # POS tags
    }
    _, vocab = conllu_reader.to_int_and_vocab(col_name_dict)
    wordvocab = vocab['form']
    tagvocab = vocab['upos']
    
    words, tags = [], []
    

    return words, tags, wordvocab, tagvocab


def read_corpus_class(filename, wordvocab, tagvocab, in_type, train_mode=True, batch_mode=True):
  if train_mode :
    wordvocab = defaultdict(lambda : len(wordvocab))
    tagvocab = defaultdict(lambda : len(tagvocab))
  words, tags = [], []
  with open(filename, 'r', encoding="utf-8") as corpus:
    for line in corpus:
      fields = line.strip().split()
      tags.append(tagvocab[fields[0]])
      fields = " ".join(fields[1:]) if in_type == "char" else  fields[1:]
      if train_mode :
        words.append([wordvocab[w] for w in fields])
      else :
        words.append([wordvocab.get(w, wordvocab["<UNK>"]) for w in fields])
  if batch_mode :
    dataset = TensorDataset(pad_tensor(words, 40), torch.LongTensor(tags))
    return DataLoader(dataset, batch_size=32, shuffle=train_mode), wordvocab, tagvocab 
  else :
    return words, tags, wordvocab, tagvocab
    




# if __name__ == "__main__" : 
#   if len(sys.argv) != 5 or \
#      sys.argv[3] not in ['bow', 'gru', 'cnn'] or \
#      sys.argv[4] not in ['word', 'char'] : # Argparse!
#     print("Usage: {} trainfile.txt devfile.txt bow|gru|cnn word|char".format(sys.argv[0]), file=sys.stderr) 
#     sys.exit(-1)   
#   hp = {"model_type": sys.argv[3], "in_type": sys.argv[4], "d_embed": 250, "d_hidden": 200}
#   train_loader, wordvocab, tagvocab = read_corpus(sys.argv[1], None, None, hp["in_type"])
#   dev_loader, _, _ = read_corpus(sys.argv[2], wordvocab, tagvocab, hp["in_type"], train_mode=False)
#   if hp["model_type"] == "bow" :
#     model = BOWClassifier(hp["d_embed"], len(wordvocab), len(tagvocab))
#   elif hp["model_type"] == "gru" :
#     model = GRUClassifier(hp["d_embed"], hp["d_hidden"], len(wordvocab), len(tagvocab))
#   else: #if hp["model_type"] == "cnn" :
#     model = CNNClassifier(hp["d_embed"], hp["d_hidden"], len(wordvocab), len(tagvocab))
#   fit(model, 15, train_loader, dev_loader)
#   torch.save({"wordvocab": dict(wordvocab), 
#               "tagvocab": dict(tagvocab), 
#               "model_params": model.state_dict(),
#               "hyperparams": hp}, "model.pt")
