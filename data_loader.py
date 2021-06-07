from torch.utils.data import Dataset, DataLoader
import os
import random
from make_vocab import *
import torch 
import numpy as np
from training_setup import *

text_path = "/ssd_scratch/cvit/seshadri_c/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual"

train_text_path = os.path.join(text_path,"PHOENIX-2014-T.train.corpus.csv")

dict_word_int, dict_int_word = build_vocab(train_text_path)
	
class DataGenerator(Dataset):
	
	def __init__(self, path):
		self.files = self.get_files(path)
        

	def __len__(self):
		return len(self.files)
        

	def __getitem__(self,idx):

		video_feature_path, cap = self.files[idx]		
		data = np.load(video_feature_path)
		return data, cap
		
			
	def get_files(self,path):

		print(path)
		files = open(path[1], 'r').read().splitlines()
		video_cap_pair =[]
		for cap in files:
			video_cap_pair.append((os.path.join(path[0], cap.split('|')[0], "Mixed_4f.npy"), cap.split('|')[-1]))
		
		return video_cap_pair[1:]

#Tokenizing the batch of sentences and adding BOS_WORD and EOS_WORD 
def tokenize_and_add_BOS_EOS(sentence):

	#Beginning of Sentence
	BOS_WORD = '<s>'
	#End of Sentence
	EOS_WORD = '</s>'

	token_list = []
	token_list.append(BOS_WORD)
	token_list.extend(tokenize_de(str(sentence)))
	token_list.append(EOS_WORD)
	
	return token_list
	

def word_to_int(sent):
	
	int_sent = []
	for t in sent:
		try:
			int_sent.append(dict_word_int[t])
		except:
			int_sent.append(2)
	return int_sent
	
def collate_fn_customised(data):
	
	vid_feature = data[0][0]
	cap = data[0][1]
	
	de_sent = tokenize_and_add_BOS_EOS(cap)
	int_sent = word_to_int(de_sent)
	
	vid_feature = np.array(vid_feature)
	vid_feature = np.transpose(vid_feature,(0, 2, 1, 3, 4))
	vid_feature = np.reshape(vid_feature,(1, vid_feature.shape[1], 163072))
	int_sent = np.array(int_sent)
	
	vid_tensor = torch.tensor(vid_feature)
	cap_tensor = torch.tensor(int_sent).unsqueeze(0)
	
	pad = 2
	src_mask, tgt_mask = make_std_mask(vid_tensor, cap_tensor, pad)
	
	#src_mask = torch.tensor(src_mask)
	#tgt_mask = torch.tensor(tgt_mask)
	
	return vid_tensor, cap_tensor, src_mask, tgt_mask
	
def load_data(data_path, batch_size=1, num_workers=10, shuffle=True):

	dataset = DataGenerator(data_path)
	data_loader = DataLoader(dataset, collate_fn = collate_fn_customised, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

	return data_loader

#data_path = "./data/multi30k/uncompressed_data"
#load_data(data_path)
