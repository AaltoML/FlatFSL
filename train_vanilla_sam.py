"""
train feature extractor with vanilla sam
"""

#######################################################
# load helper functions and libraries
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import os
from tqdm import tqdm

import numpy as np
import argparse
from models.resnet18 import create_resnet18
from models.losses import cross_entropy_loss, prototype_loss

from models.sam import SAM
from util_function import enable_running_stats, disable_running_stats

parser = argparse.ArgumentParser('train feature extractor with sam')
parser.add_argument("--eval_size", default=300, type = int)
parser.add_argument("--job_id", default=0, type = int)
parser.add_argument("--dataset_id", default=0, type = int)
parser.add_argument("--weight_decay", default=0.0007, type = float)
args = parser.parse_args()

#######################################################
# set hyperparameters
dataset_list = "ilsvrc_2012 omniglot aircraft cu_birds dtd quickdraw fungi vgg_flower".split(' ')

trainsets = [dataset_list[args.dataset_id]]
train_dataset = dataset_list[args.dataset_id]

valsets = trainsets
hyparam_list = {"ilsvrc_2012": [64, 3e-2, 480000, 48000, 0.05], 
                'omniglot': [16, 3e-2, 50000, 3000, 0.01], 
                'aircraft':[8, 3e-2, 50000, 3000, 0.1],
                'cu_birds': [16, 3e-2, 50000, 3000, 0.1],
                'dtd': [32, 3e-2, 50000, 1500, 0.05],
                'quickdraw': [64, 1e-2, 480000, 48000, 0.05],
                'fungi': [32, 3e-2, 480000, 15000, 0.05],
                'vgg_flower': [8, 3e-2, 50000, 1500, 0.1]}

batch_size, lr, iter_num, optimizer_restart, rho = hyparam_list[train_dataset]
eval_every = optimizer_restart
weight_decay = args.weight_decay
eval_size = args.eval_size

#######################################################
# data set loader
from meta_dataset import MetaDatasetEpisodeReader, MetaDatasetBatchReader
testsets = ['mnist']

batch_loader = MetaDatasetBatchReader('train', trainsets, trainsets, testsets, batch_size=batch_size)
val_loader = MetaDatasetEpisodeReader('val', trainsets, valsets, testsets)

num_train_classes = batch_loader.num_classes('train')

#######################################################
# create model
model = create_resnet18(num_train_classes).cuda()

#######################################################
# start traning
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = False
session = tf.compat.v1.Session(config=config)

base_optimizer = torch.optim.SGD
optimizer = SAM(model.parameters(), base_optimizer, lr = lr, rho = rho, weight_decay = weight_decay, momentum=0.9, nesterov=False)
lr_manager = CosineAnnealingWarmRestarts(optimizer.base_optimizer, optimizer_restart)

best_val_acc = 0
for iteration in tqdm(range(iter_num)):
    
    model.train()
    
    ################################### train ###################################
    batch_data = batch_loader.get_train_batch(session)
    
    # first forward-backward step
    enable_running_stats(model)
    pred = model.forward(batch_data['images'])
    loss, _ = cross_entropy_loss(pred, batch_data['labels'])
    loss.backward()
    optimizer.first_step(zero_grad=True)
    
    # second forward-backward step
    disable_running_stats(model)
    pred = model.forward(batch_data['images'])
    loss, _ = cross_entropy_loss(pred, batch_data['labels'])
    loss.backward()
    optimizer.second_step(zero_grad=True)
    
    with torch.no_grad():
        sl_loss, sl_acc = cross_entropy_loss(pred, batch_data['labels'])
        
    # learniing rate scheduler
    lr_manager.step(iteration)

    ################################### validation ###################################
    if (iteration + 1) % eval_every == 0:
        model.eval()
        total_val_acc = []
        for valset in valsets:
            dataset_acc = []
            for j in tqdm(range(eval_size)):
                episode_data = val_loader.get_validation_task(session, valset)
                
                with torch.no_grad():
                    context_features = model.embed(episode_data['context_images'].to(device))
                    target_features = model.embed(episode_data['target_images'].to(device))     

                fsc_loss, fsc_acc = prototype_loss(context_features, episode_data['context_labels'],
                                                    target_features, episode_data['target_labels'], distance = 'cos')
                
                dataset_acc.append(fsc_acc.cpu().item())
            
            total_val_acc.append(np.mean(dataset_acc))
        
        if np.mean(total_val_acc) > best_val_acc:
            best_val_acc = np.mean(total_val_acc)
            print(f"--------------------------------- best model at {iteration}, val acc {best_val_acc} ---------------------------------")
            state = {'epoch': iteration + 1, 'state_dict': model.get_state_dict()} 
            torch.save(state, os.path.join(f"saved_model/sam/vanilla/{train_dataset}.pth"))