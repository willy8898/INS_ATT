import os
from os.path import join
from glob import glob
import yaml
import time
import argparse

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb

######  The network and loss are figured out here  ###### 
from chamferdist import ChamferDistance
from ins_embed import discriminative_loss
###### ------ ######
from network_ins_aux import Net_conpu_v7

#############################################################################################################################

#############################################################################################################################

class KITTI_Patch_Dataset(Dataset):
    def __init__(self, path, unify_back, set, balance_stuff=False):
        # path contain 'scene_patch_1024.npy' and 'scene_patch_256.npy'
        self.path = path
        self.dense_path = join(path, 'subscene_dense')
        self.sparse_path = join(path, 'subscene')
        self.balance_stuff = balance_stuff
        self.stuff_dense_path = join(path, 'subscene_dense_stuff')
        self.stuff_sparse_path = join(path, 'subscene_stuff')

        if unify_back:
            self.ins_mask_path = join(path, 'ins_mask')
            self.lbl_path = join(path, 'label')
            self.stuff_mask_path = join(path, 'ins_mask_stuff')
            self.stuff_lbl_path = join(path, 'label_stuff')
        else: # no unify background
            self.ins_mask_path = join(path, 'ins_mask_origin')
            self.lbl_path = join(path, 'label_origin')
            self.stuff_mask_path = join(path, 'ins_mask_origin_stuff')
            self.stuff_lbl_path = join(path, 'label_origin_stuff')

        self.set = set
        #self.batch_size = batch_size

        # create training file list
        if self.set == 'train':
            self.sequences = ['00']
        elif self.set == 'validation':
            self.sequences = ['10']
        self.load_data(self.sequences)

        with open('semantic-kitti.yaml', 'r') as stream:
            doc = yaml.safe_load(stream)
            learning_map = doc['learning_map']
            self.learning_map = np.zeros((np.max([k for k in learning_map.keys()]) + 1), dtype=np.int32)
            for k, v in learning_map.items():
                self.learning_map[k] = v

    def __len__(self):
        return len(self.sparse_list)

    def __getitem__(self, idx):
        dense_pts = np.load(self.dense_list[idx])
        sparse_pts = np.load(self.sparse_list[idx])
        ins_mask = np.load(self.ins_mask_list[idx])
        lbl = np.load(self.lbl_list[idx])
        sem_labels = lbl & 0xFFFF  # semantic label in lower half
        sem_labels = self.learning_map[sem_labels]

        return dense_pts, sparse_pts, ins_mask, sem_labels
        #nor_dense_pts, nor_sparse_pts = self.normalize(dense_pts, sparse_pts)
        #return nor_dense_pts, nor_sparse_pts, sparse_pts, ins_mask, lbl
    
    def load_data(self, sequences):
        '''
        Create file list
        '''
        self.dense_list = []
        self.sparse_list = []
        self.ins_mask_list = []
        self.lbl_list = []
        for seq_i in sequences:
            self.dense_list.extend(
                sorted(glob(join(self.dense_path, seq_i, '*.npy'))))
            self.sparse_list.extend(
                sorted(glob(join(self.sparse_path, seq_i, '*.npy'))))
            self.ins_mask_list.extend(
                sorted(glob(join(self.ins_mask_path, seq_i, '*.npy'))))
            self.lbl_list.extend(
                sorted(glob(join(self.lbl_path, seq_i, '*.npy'))))
            if self.balance_stuff:
                self.dense_list.extend(
                    sorted(glob(join(self.stuff_dense_path, seq_i, '*.npy'))))
                self.sparse_list.extend(
                    sorted(glob(join(self.stuff_sparse_path, seq_i, '*.npy'))))
                self.ins_mask_list.extend(
                    sorted(glob(join(self.stuff_mask_path, seq_i, '*.npy'))))
                self.lbl_list.extend(
                    sorted(glob(join(self.stuff_lbl_path, seq_i, '*.npy'))))
                
        
    def normalize(self, dense, sparse):
        '''
        - dense, sparse: [N, 3]
        Normalize point data (range: -1 ~ 1)
        '''
        centroid = np.mean(sparse, axis=0, keepdims=True)
        sparse = sparse - centroid
        furthest_distance = np.amax(np.sqrt(np.sum(sparse ** 2, axis=-1)), axis=0, keepdims=True)
        sparse = sparse / np.expand_dims(furthest_distance, axis=-1)

        dense = dense - centroid
        dense = dense / np.expand_dims(furthest_distance, axis=-1)

        return dense, sparse

#############################################################################################################################

#############################################################################################################################

def chf_loss(upsampled_pts, gt_pts):
    # upsampled_pts: (B, N, 3)
    # gt_pts: (B, N, 3)
    chamferDist = ChamferDistance()
    forward = chamferDist(upsampled_pts, gt_pts, reduction=None)
    backward = chamferDist(gt_pts, upsampled_pts, reduction=None)
    chamfer_loss = forward.mean() + backward.mean()
    return chamfer_loss

def train_net(net, train_loader, val_loader, optimizer, scheduler, args, use_wandb=False):
    ################
    # wandb
    ################
    if use_wandb:
        wandb.init(config=args,
                   project='npt_ins_aux',
                   name=time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime()))

    max_epoch = args.max_epoch
    best_loss = 100
    best_val_loss = 100

    for epoch_i in range(max_epoch):
        print(f'Epoch {epoch_i} training...')
        epoch_loss, epoch_chfloss, epoch_ins_loss = train_epoch(net, train_loader, optimizer, scheduler, args, use_wandb=use_wandb)
        if use_wandb:
            wandb.log({'epoch_loss': epoch_loss, 'epoch_chfloss': epoch_chfloss, 'epoch_ins_loss': epoch_ins_loss})
        print(f'epoch_{epoch_i} loss: {epoch_loss:10.6f}')

        if epoch_loss < best_loss:
            saving_ckpt_path = join(args.saving_path, 'best_loss_ckpt.pt')
            torch.save(net.state_dict(), saving_ckpt_path)
        if epoch_i % 5 == 0:
            saving_ckpt_path = join(args.saving_path, f'epoch_{epoch_i}_ckpt.pt')
            torch.save(net.state_dict(), saving_ckpt_path)

        # validate
        if (epoch_i >= 0) and (epoch_i % 5 == 0):
            print('start validation...')
            val_loss, val_chfloss, val_ins_loss = val_epoch(net, val_loader, args)
            if val_loss < best_val_loss:
                saving_ckpt_path = join(args.saving_path, f'best_val_loss_ckpt.pt')
                torch.save(net.state_dict(), saving_ckpt_path)
            print(f'epoch_{epoch_i} val loss: {val_loss:10.6f}')
            if use_wandb:
                wandb.log({'val_loss': val_loss, 'val_chfloss': val_chfloss, 'val_ins_loss': val_ins_loss})
    return

def train_epoch(net, dataloader, optimizer, scheduler, args, use_wandb=False):
    step = 0
    epoch_loss = 0
    epoch_chfloss = 0
    epoch_ins_loss = 0
    net.train()

    for dense, sparse, target, sem_info in tqdm(dataloader, total=len(dataloader)):
        sparse = sparse.cuda()
        dense = dense.cuda()
        if args.back_loss:
            target = target.cuda()
        else:
            target = target[:, :, 1:].cuda()
        # We use semantic information in training stage, but found it no help for upsampling
        # Thus, in inference stage, sem info can set to zero tensor and won't affect performance largely.
        sem_info = sem_info.view(sparse.shape[0], sparse.shape[1], 1).type(torch.FloatTensor).cuda()
        n_obj = [target.shape[2] for _ in range(len(target))]

        optimizer.zero_grad()

        gen_points_batch, ins_embed = net(sparse, sem_info)
        chfloss = chf_loss(gen_points_batch, dense)
        ins_loss = discriminative_loss(ins_embed, target, n_objects=n_obj,
                        max_n_objects=64, delta_v=args.delta_v, delta_d=args.delta_d, norm=2, usegpu=True)
        if torch.isnan(ins_loss).any():
            epoch_ins_loss += 0
            loss = 100*chfloss
        else:
            epoch_ins_loss += ins_loss.item()
            loss = 100*chfloss + ins_loss
        epoch_chfloss += chfloss.item()
        epoch_loss += loss.item()

        net.zero_grad()
        # Backward + optimize
        loss.backward()  
        torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
        optimizer.step()
        scheduler.step()
        if use_wandb:
            wandb.log({
                'step loss': loss.item(), 'step chfloss': chfloss.item(), 'step ins_loss': ins_loss.item()})

        step += 1

    return epoch_loss/step, epoch_chfloss/step, epoch_ins_loss/step

def val_epoch(net, dataloader, args):
    step = 0
    epoch_loss = 0
    epoch_chfloss = 0
    epoch_ins_loss = 0
    net.eval()
    with torch.no_grad():
        for dense, sparse, target, sem_info in tqdm(dataloader, total=len(dataloader)):
            sparse = sparse.cuda()
            dense = dense.cuda()
            if args.back_loss:
                target = target.cuda()
            else:
                target = target[:, :, 1:].cuda()
            sem_info = sem_info.view(sparse.shape[0], sparse.shape[1], 1).type(torch.FloatTensor).cuda()
            n_obj = [target.shape[2] for _ in range(len(target))]

            gen_points_batch, ins_embed = net(sparse, sem_info)
            chfloss = chf_loss(gen_points_batch, dense)
            ins_loss = discriminative_loss(ins_embed, target, n_objects=n_obj,
                            max_n_objects=64, delta_v=args.delta_v, delta_d=args.delta_d, norm=2, usegpu=True)

            if torch.isnan(ins_loss).any():
                epoch_ins_loss += 0
                loss = 100*chfloss
            else:
                epoch_ins_loss += ins_loss.item()
                loss = 100*chfloss + ins_loss
            epoch_chfloss += chfloss.item()
            epoch_loss += loss.item()
            step += 1
    
    return epoch_loss/step, epoch_chfloss/step, epoch_ins_loss/step


if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_point', type=int, default=256,help='Point Number')
    parser.add_argument('--training_up_ratio', type=int, default=4,help='The Upsampling Ratio during training') 
    parser.add_argument('--testing_up_ratio', type=int, default=4, help='The Upsampling Ratio during testing')  
    parser.add_argument('--over_sampling_scale', type=float, default=1.5, help='The scale for over-sampling')
    parser.add_argument('--emb_dims', type=int, default=512, metavar='N',help='Dimension of embeddings')
    parser.add_argument('--pe_out_L', type=int, default=5, metavar='N',help='The parameter L in the position code')
    parser.add_argument('--feature_unfolding_nei_num', type=int, default=4, metavar='N',help='The number of neighbour points used while feature unfolding')

    # for phase train
    parser.add_argument('--data_path', type=str, default='/media/1TB_HDD/kitti_trainset/kitti_ins_balance/', help='where your processed dataset is')
    parser.add_argument('--max_epoch', type=int, default=100, help='Epoch to run')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--if_bn', type=int, default=0, help='If using batch normalization')
    parser.add_argument('--neighbor_k', type=int, default=10, help='The number of neighbour points used in DGCNN')
    parser.add_argument('--mlp_fitting_str', type=str, default='256 128 64', metavar='None',help='mlp layers of the part surface fitting (default: None)')

    parser.add_argument('--weight_decay',default=0.00005, type=float)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)

    # control the using mode
    parser.add_argument('--if_fix_sample', type=int, default=0, help='whether to use fix sampling')
    parser.add_argument('--if_use_siren', type=int, default=0, help='whether to use siren activation function')
    parser.add_argument('--saving_path', type=str, default='result', help='ckpt saving path')

    parser.add_argument('--pretrain', type=str, default=None, help='pretrained ckpt path')
    # if unify background ( call --unify_back --back_loss)
    # default: no unify background + calculate the background embedding point loss
    parser.add_argument('--unify_back', action='store_true', default=False, help='if unify background')
    parser.add_argument('--back_loss', action='store_false', default=True, help='calculate background loss')
    parser.add_argument('--balance_stuff', action='store_true', default=False, help='add stuff scene')
    # ins loss dist
    parser.add_argument('--delta_v', type=float, default=1.0, help='ins loss delta_v')
    parser.add_argument('--delta_d', type=float, default=2.5, help='ins loss delta_d')
    parser.add_argument('--chf_weight', type=float, default=100.0, help='chf loss weight')
    # log
    parser.add_argument('--use_wandb', type=bool, default=False, help='if use wandb to record')
    args = parser.parse_args()

    net = Net_conpu_v7(args).cuda()
    if args.pretrain is not None: # restore from pretrained
        net.load_state_dict(torch.load(args.pretrain))

    print('prepare dataset .....')
    # dataset
    # Initialize datasets
    path = args.data_path
    training_dataset = KITTI_Patch_Dataset(path, args.unify_back, set='train', balance_stuff=args.balance_stuff)
    val_dataset = KITTI_Patch_Dataset(path, args.unify_back, set='validation', balance_stuff=args.balance_stuff)
    print(f'Total training data: {len(training_dataset)}')
    print(f'Total validation data: {len(val_dataset)}')

    # Initialize the dataloader
    train_loader = DataLoader(training_dataset,
                              batch_size=1,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True)

    # setup optimizer
    optimizer = optim.AdamW(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, eps=args.epsilon)
    scheduler_step = len(train_loader)*50
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, 
                                              max_lr=args.learning_rate,
                                              total_steps=scheduler_step, 
                                              pct_start=0.03, 
                                              cycle_momentum=False, 
                                              anneal_strategy='linear')

    print('start training...')
    # run train and tvalidation
    train_net(net, train_loader, val_loader, optimizer, scheduler, args, use_wandb=args.use_wandb)
    print('Done.')