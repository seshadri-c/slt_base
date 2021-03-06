
from header import *

class Src_Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Src_Embeddings, self).__init__()
        self.lut = nn.Linear(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
        
class Tgt_Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Tgt_Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
