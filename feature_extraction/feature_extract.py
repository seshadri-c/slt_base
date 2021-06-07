import os
from data_loader_fe import *
from tqdm import tqdm

from pytorch_i3d import InceptionI3d
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))
device = torch.device("cuda" if use_cuda else "cpu")  

#SETTING UP DATA PATH
video_path = "/ssd_scratch/cvit/seshadri_c/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px"

train_video_path = os.path.join(video_path,"train")
dev_video_path = os.path.join(video_path,"dev")
test_video_path = os.path.join(video_path,"test")

def feature_extract(data_loader,path,model):

	progress_bar = tqdm(enumerate(data_loader))
	
	for step,(vid_addr, img_frames) in progress_bar:
		
		try :
			with torch.no_grad():
				
				data = Variable(img_frames.to(device)).float()
				features = model.extract_features(data)
				video_path = os.path.join(path,vid_addr[0])
				os.mkdir(video_path)
				
				for f in features:
					feature_path = os.path.join(video_path,f[0]+".npy")
					with open(feature_path, 'wb') as fp:
						np.save(fp, f[1].data.cpu().numpy())
						
				print("Video : ",vid_addr)
		except :
			continue
		
def main():
	
	#train_loader = load_data(train_video_path, batch_size=1, num_workers=10, shuffle=True)
	#test_loader = load_data(test_video_path, batch_size=1, num_workers=10, shuffle=True)
	dev_loader = load_data(dev_video_path, batch_size=1, num_workers=10, shuffle=True)
	
	load_model = "models/rgb_imagenet.pt"
	model = InceptionI3d(400, in_channels=3,final_endpoint='Logits')
	model.load_state_dict(torch.load(load_model))
	model.to(device)
	model.train(False)
	
	#train_feature_path = "/ssd_scratch/cvit/seshadri_c/phoenix_features/train"
	#test_feature_path = "/ssd_scratch/cvit/seshadri_c/phoenix_features/test"
	dev_feature_path = "/ssd_scratch/cvit/seshadri_c/phoenix_features/dev"
	
	#feature_extract(train_loader,train_feature_path,model)
	#feature_extract(test_loader,test_feature_path,model)
	feature_extract(dev_loader,dev_feature_path,model)
	
main()
