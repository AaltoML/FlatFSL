import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import torch
from models.resnet18 import create_resnet18
import copy
import numpy as np
import scipy
from tqdm import tqdm

# data set loader
from meta_dataset import MetaDatasetEpisodeReader
trainsets = "ilsvrc_2012 omniglot aircraft cu_birds dtd quickdraw fungi vgg_flower".split(' ')
testsets = "traffic_sign mnist mscoco cifar10 cifar100".split(' ')
test_loader = MetaDatasetEpisodeReader('test', trainsets, trainsets, testsets, test_type='standard')

# pretrained model
pretrained_models = {}

model = create_resnet18(None)
for dataset in trainsets:
    checkpoint = torch.load(f"saved_model/sam/{dataset}.pth")
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    pretrained_models[dataset] = copy.deepcopy(model).cuda()

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = False
session = tf.compat.v1.Session(config=config)

def get_lowertri(rdm):
    num_conditions = rdm.shape[0]
    return rdm[np.triu_indices(num_conditions,1)]

pretrained_backbone_score = {}

for test_dataset in testsets:
    cur_score = {dataset:[] for dataset in trainsets}
    for i in tqdm(range(1)):
        episode_data = test_loader.get_test_task(session, test_dataset)
        one_hot_label = torch.nn.functional.one_hot(episode_data['context_labels'])
        D_labels = 1 - np.corrcoef(one_hot_label.cpu())

        for train_dataset in trainsets:
            features = pretrained_models[train_dataset].embed(episode_data['context_images'])
            D_feautres = 1 - np.corrcoef(features.detach().cpu())

            score = scipy.stats.spearmanr(get_lowertri(D_feautres), get_lowertri(D_labels))[0]
            
            cur_score[train_dataset].append(score)
        
    pretrained_backbone_score[test_dataset] = cur_score

for test_dataset in testsets:
    print(f"------------------------------- {test_dataset} -------------------------------")
    for train_dataset in trainsets:
        cur_score= pretrained_backbone_score[test_dataset][train_dataset]
        print(f"{train_dataset}: {np.mean(cur_score):.2f} ± {np.std(cur_score):.2f}")



