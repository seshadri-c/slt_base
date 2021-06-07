from data_loader import *
from tqdm import tqdm
import numpy as np
import spacy
from make_transformer_model import *
from optimizer import *
from label_smoothing import *
from training_setup import *
from torchtext import data
from make_vocab import *
import random
from torchtext.data.metrics import bleu_score
	
use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))
device = torch.device("cuda" if use_cuda else "cpu")

#Function to Save Checkpoint
def save_ckp(checkpoint, checkpoint_path):
    torch.save(checkpoint, checkpoint_path)

#Function to Load Checkpoint
def load_ckp(checkpoint_path, model, model_opt):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    model_opt.optimizer.load_state_dict(checkpoint['optimizer'])
    return model, model_opt, checkpoint['epoch']
    
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    
    temp = torch.tensor([0], dtype=torch.long, device=device)
    
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(temp.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                           Variable(ys), 
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(temp.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(temp.data).fill_(next_word)], dim=1)
    return ys
    
def decode_target(cap_tensor):
	
	tgt = ""
	for t in np.array(cap_tensor.data.squeeze(0)):
		sym = dict_int_word[t]
		if sym == "</s>": 
			break
		if sym == "<s>":
			continue
		tgt += sym + " "
	return tgt
	
def test_epoch(epoch,valid_loader, model, criterion):
	
	progress_bar = tqdm(enumerate(valid_loader))
	model.eval()
	
	target_list = []
	predicted_list = []
	j=0
	
	for step, (video_tensor, cap_tensor, src_mask, tgt_mask) in progress_bar:
		out = greedy_decode(model, video_tensor.to(device), src_mask.cuda(), max_len=60, start_symbol=dict_word_int["<s>"])
		
		trans = ""
		for i in range(1, out.size(1)):
			sym = dict_int_word[int(out[0, i])]
			if sym == "</s>": 
				break
			trans += sym + " "
			
		target_list.append([decode_target(cap_tensor).upper().split()])
		predicted_list.append(trans.upper().split())
		print("\n\n Pair : {}\n Target : {} \n Predicted : {}".format(j+1, target_list[-1], predicted_list[-1]))
		print("The BLEU Score : ",bleu_score(predicted_list, target_list)*100,"\n\n")
		j+=1
		
def	training_testing(test_loader, model, criterion, model_opt, num_epochs, resume, dict_int_word):
	
	checkpoint_dir = "/ssd_scratch/cvit/seshadri_c/TRAIN_1/checkpoints/"
	checkpoint_path_latest = checkpoint_dir + "checkpoint_latest.pt"
	model, model_opt, epoch = load_ckp(checkpoint_path_latest, model, model_opt)
	print("Model loaded succesfully from : ", checkpoint_path_latest)
	test_epoch(epoch,test_loader, model, criterion)	
		
		
def main():
	
	video_features_path = "/ssd_scratch/cvit/seshadri_c/phoenix_features"
	text_path = "/ssd_scratch/cvit/seshadri_c/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual"
	
	train_video_path = os.path.join(video_features_path,"train")
	dev_video_path = os.path.join(video_features_path,"dev")
	test_video_path = os.path.join(video_features_path,"test")
	
	train_text_path = os.path.join(text_path,"PHOENIX-2014-T.train.corpus.csv")
	dev_text_path = os.path.join(text_path,"PHOENIX-2014-T.dev.corpus.csv")
	test_text_path = os.path.join(text_path,"PHOENIX-2014-T.test.corpus.csv")
	
	dict_word_int, dict_int_word = build_vocab(train_text_path)
	
	#Model made only with Train Vocab data	
	model = make_model(163072, len(dict_word_int.keys()), N=6)
	model.to(device)
	model_opt = get_std_opt(model)
	
	
	#Input is the Target Vocab Size
	criterion = LabelSmoothing(size=len(dict_word_int.keys()), padding_idx=2, smoothing=0.1)
	criterion.to(device)
	
	train_path = (train_video_path, train_text_path)
	dev_path = (dev_video_path, dev_text_path)
	test_path = (test_video_path, test_text_path)
	
	
	test_loader = load_data(train_path, batch_size=1, num_workers=10, shuffle=True)

	num_epochs = 20
	resume = True
	training_testing(test_loader, model, criterion, model_opt, num_epochs, resume, dict_int_word)	
main()
