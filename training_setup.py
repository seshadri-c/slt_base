from torch.autograd import Variable
from transformer.mask import *
from loss_backprop import *

def make_std_mask(src, tgt, pad):
    "Create a mask to hide padding and future words."
    src_mask = []
    for i in range(src.shape[1]):
        src_mask.append(True)
    src_mask = torch.tensor(np.array(src_mask)).unsqueeze(-2)
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(
    subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return src_mask, tgt_mask
