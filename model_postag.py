import torch 
import torch.nn as nn


class RNN_postag(nn.Module):

    def __init__(self, hidden_size: int, output_size: int, num_embeddings: int, embedding_dim: int, PAD_ID: int):
        super().__init__()
        self.PAD_ID = PAD_ID
        self.embed = nn.Embedding(num_embeddings, embedding_dim, padding_idx=PAD_ID)
        self.gru = nn.GRU(embedding_dim, hidden_size, batch_first=True, bias=False)
        self.dropout = nn.Dropout(0.1)
        self.decision = nn.Linear(hidden_size, output_size)

    
    def forward(self, idx_words):
        embedding = self.embed(idx_words)
        # hidden = self.gru(embedding)[1].squeeze(dim=0)
        hidden = self.gru(embedding)[1].squeeze(dim=0)
        return self.decision(self.dropout(hidden))