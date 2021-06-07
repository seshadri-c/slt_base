#IMPORT HEADERS
import os
import cv2
import numpy as np
import torch 

from torch.utils.data import Dataset, DataLoader

class PHOENIX(Dataset):
	
	def __init__(self, path):
		self.files = self.get_files(path)
        

	def __len__(self):
		return len(self.files)
        

	def __getitem__(self,idx):

		vid_addr = self.files[idx]
		img_list = [img for img in os.listdir(vid_addr)]
		img_list.sort()
		image_frames = []
		for i_n in img_list:
			img = cv2.imread(os.path.join(vid_addr, i_n))
			if(img is not None):
				img = cv2.resize(img,(224,224))
				data = np.array(img)
				data = data.astype(float)
				data = (data * 2 / 255) - 1
				image_frames.append(data)
		image_frames = np.array(image_frames)
		image_frames = image_frames.transpose((3,0,1,2))
		return (vid_addr.split('/')[-1], image_frames)
		
	def get_files(self, path):

		vid_addr = []
		vid_addr = [os.path.join(path,v) for v in os.listdir(path)]
		return vid_addr
		
def load_data(data_path, batch_size, num_workers=10, shuffle=True):
    dataset = PHOENIX(data_path)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=10, shuffle=True)
    return data_loader
    

