import copy
import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../code')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import grad

from pointnet2 import pointnet2_utils as pn2_utils
from attention_module import LocalTransformer, Feature_MLPNet_relu
import torch_tensor_functions

######## TODO: START PART: FUNCTIONS ABOUT DGCNN. IT IS USED AS THE FEATURE EXTRACTOR IN OUR FRAMEWORK. ########
#### The DGCNN network ####
class DGCNN_multi_knn_c5(nn.Module):
    def __init__(self, emb_dims=512, args=None):
        super(DGCNN_multi_knn_c5, self).__init__()
        self.args = args
        self.conv1 = nn.Conv2d(6, 64, kernel_size=1, bias=False)
        init.xavier_normal_(self.conv1.weight, gain=1.0)
        self.conv2 = nn.Conv2d(64*2, 64, kernel_size=1, bias=False)
        init.xavier_normal_(self.conv2.weight, gain=1.0)
        self.conv3 = nn.Conv2d(64*2, 128, kernel_size=1, bias=False)
        init.xavier_normal_(self.conv3.weight, gain=1.0)
        self.conv4 = nn.Conv2d(128*2, 256, kernel_size=1, bias=False)
        init.xavier_normal_(self.conv4.weight, gain=1.0)
        self.conv5 = nn.Conv2d(512, emb_dims, kernel_size=1, bias=False)
        init.xavier_normal_(self.conv5.weight, gain=1.0)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(emb_dims)
    def forward(self, x, if_relu_atlast = False):
        batch_size, num_dims, num_points = x.size()
        x = get_graph_feature(x) # This sub model get the graph-based features for the following 2D convs
        # The x is similar with 2D image
        if self.args.if_bn == True: x = F.relu(self.bn1(self.conv1(x)))
        else: x = F.relu(self.conv1(x))
        x1 = x.max(dim=-1, keepdim=False)[0]
        x = get_graph_feature(x1)
        if self.args.if_bn == True: x = F.relu(self.bn2(self.conv2(x))) 
        else: x = F.relu(self.conv2(x))
        x2 = x.max(dim=-1, keepdim=False)[0]
        x = get_graph_feature(x2)
        if self.args.if_bn == True: x = F.relu(self.bn3(self.conv3(x))) 
        else: x = F.relu(self.conv3(x))
        x3 = x.max(dim=-1, keepdim=False)[0]
        x = get_graph_feature(x3)
        if self.args.if_bn == True: x = F.relu(self.bn4(self.conv4(x))) 
        else: x = F.relu(self.conv4(x))
        x4 = x.max(dim=-1, keepdim=False)[0]
        x = torch.cat((x1, x2, x3, x4), dim=1).unsqueeze(3)
        if if_relu_atlast == False:
            return torch.tanh(self.conv5(x)).view(batch_size, -1, num_points)
        x = F.relu(self.conv5(x)).view(batch_size, -1, num_points)
        return x
#### The knn function used in graph_feature ####
def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx
#### The edge_feature used in DGCNN ####
def get_graph_feature(x, k=4):
    idx = knn(x, k=k)  # (batch_size, num_points, k)
    batch_size, num_points, _ = idx.size()
    device = torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2,1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)
    return feature
######## TODO: END PART: FUNCTIONS ABOUT DGCNN. IT IS USED AS THE FEATURE EXTRACTOR IN OUR FRAMEWORK. ########

######## TODO: START PART: NEURAL IMPLICIT FUNCTION, MLP with ReLU. ########
#### Construct the neural implicit function. ####
class MLPNet_relu(torch.nn.Module):
    """ Multi-layer perception.
        [B, Cin, N] -> [B, Cout, N] or
        [B, Cin] -> [B, Cout]
    """
    def __init__(self, nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0, if_bn = True):
        super().__init__()
        list_layers = mlp_layers_relu(nch_input, nch_layers, b_shared, bn_momentum, dropout, if_bn)
        self.layers = torch.nn.Sequential(*list_layers)
    def forward(self, inp):
        out = self.layers(inp)
        return out
#### Construct the mlp_layers of the neural implicit function. ####
def mlp_layers_relu(nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0, if_bn=True):
    """ [B, Cin, N] -> [B, Cout, N] or
        [B, Cin] -> [B, Cout]
    """
    layers = []
    last = nch_input
    for i, outp in enumerate(nch_layers):
        if b_shared:
            weights = torch.nn.Conv1d(last, outp, 1)
            init.xavier_normal_(weights.weight, gain=1.0)
            # if i==0: init.uniform_(weights.weight, a=-(6/last)**0.5*30, b=(6/last)**0.5*30)
            # else: init.uniform_(weights.weight, a=-(6/last)**0.5, b=(6/last)**0.5)
        else:
            weights = torch.nn.Linear(last, outp)
            init.xavier_normal_(weights.weight, gain=1.0)
        layers.append(weights)
        if if_bn==True:
            layers.append(torch.nn.BatchNorm1d(outp, momentum=bn_momentum))
        layers.append(torch.nn.ReLU())
        # layers.append(Sine())
        if b_shared == False and dropout > 0.0:
            layers.append(torch.nn.Dropout(dropout))
        last = outp
    return layers
######## TODO: END PART: NEURAL IMPLICIT FUNCTION, MLP with ReLU. ########


######## TODO: START PART: NEURAL IMPLICIT FUNCTION, MLP with SIREN. ########
#### Construct the neural implicit function. ####
class MLPNet(torch.nn.Module):
    """ Multi-layer perception.
        [B, Cin, N] -> [B, Cout, N] or
        [B, Cin] -> [B, Cout]
    """
    def __init__(self, nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0, if_bn = True):
        super().__init__()
        list_layers = mlp_layers(nch_input, nch_layers, b_shared, bn_momentum, dropout, if_bn)
        self.layers = torch.nn.Sequential(*list_layers)
    def forward(self, inp):
        out = self.layers(inp)
        return out
#### Construct the mlp_layers of the neural implicit function. ####
def mlp_layers(nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0, if_bn=True):
    """ [B, Cin, N] -> [B, Cout, N] or
        [B, Cin] -> [B, Cout]
    """
    layers = []
    last = nch_input
    for i, outp in enumerate(nch_layers):
        if b_shared:
            weights = torch.nn.Conv1d(last, outp, 1)
            #init.xavier_normal_(weights.weight, gain=1.0)
            if i==0: init.uniform_(weights.weight, a=-(6/last)**0.5*30, b=(6/last)**0.5*30)
            else: init.uniform_(weights.weight, a=-(6/last)**0.5, b=(6/last)**0.5)
        else:
            weights = torch.nn.Linear(last, outp)
            init.xavier_normal_(weights.weight, gain=1.0)
        layers.append(weights)
        if if_bn==True:
            layers.append(torch.nn.BatchNorm1d(outp, momentum=bn_momentum))
        #layers.append(torch.nn.ReLU())
        layers.append(Sine())
        if b_shared == False and dropout > 0.0:
            layers.append(torch.nn.Dropout(dropout))
        last = outp
    return layers
#### The nn.Moudle Sine, as the activation function, used in the nearal implicit function. ####
class Sine(nn.Module):
    def __init(self):
        super().__init__()
    def forward(self, input):
        return torch.sin(input)
######## TODO: END PART: NEURAL IMPLICIT FUNCTION, MLP with SIREN. ########


######## TODO: START PART: OUR OWN NETWORK ########
#### The main network ####
class Net_conpu_v7(nn.Module):
    def __init__(self, args):
        super(Net_conpu_v7, self).__init__()
        self.training = True
        # basic settings
        self.args = args
        self.emb_dims = args.emb_dims # 2*emb_dims = the DGCNN output
        self.up_ratio = -1
        self.over_sampling_up_ratio = -1
        self.mlp_fitting_str = self.args.mlp_fitting_str # Neural Field mlp dim string
        self.mlp_fitting = convert_str_2_list(self.mlp_fitting_str) # the channels of the layers in the MLP
        
        ######################## LAYERS #########################
        ## 1. The point-wise feature extraction, DGCNN.
        self.emb_nn_sparse = DGCNN_multi_knn_c5(emb_dims=self.emb_dims, args=self.args) # the DGCNN backbone, which is shared by all the local parts

        ## TO-DO: 2. Feature processing(point-former)
        self.Final_feature = LocalTransformer(128, 2*self.emb_dims, fea_in_dim=args.emb_dims*2, nhead=4, num_layers=2, drop=0.1, prenorm=True)

        ## 3.1 Ins Embedding.
        self.ins_head = Feature_MLPNet_relu(2*self.emb_dims, 64)

        ## 3.2.1 The Neural Field, MLP.
        if self.args.if_use_siren==True: self.fitting_mlp = MLPNet(2*self.emb_dims+(self.args.pe_out_L*4+2), self.mlp_fitting, b_shared=True, if_bn =False).layers
        else: self.fitting_mlp = MLPNet_relu(2*self.emb_dims+(self.args.pe_out_L*4+2), self.mlp_fitting, b_shared=True, if_bn =False).layers   
        self.reconstruct_out_p = torch.nn.Conv1d(self.mlp_fitting[-1], 3, 1)
        init.xavier_normal_(self.reconstruct_out_p.weight, gain=1.0)
        self.convert_feature_to_point_2to3 = torch.nn.Sequential(self.fitting_mlp, self.reconstruct_out_p)   # the Neural Field Fuction (MLP)

        ## 3.2.2 coordinate refinement
        #self.sr_head = Feature_MLPNet_relu(2*self.emb_dims, 2*self.emb_dims)
        self.sr_head = Feature_MLPNet_relu(4*self.emb_dims, 2*self.emb_dims, 128)
        if self.args.if_use_siren==True: self.fitting_mlp_r = MLPNet(2*self.emb_dims+(self.args.pe_out_L*4+2), self.mlp_fitting, b_shared=True, if_bn =False).layers
        else: self.fitting_mlp_r = MLPNet_relu(2*self.emb_dims+(self.args.pe_out_L*4+2), self.mlp_fitting, b_shared=True, if_bn =False).layers   
        self.reconstruct_out_p_r = torch.nn.Conv1d(self.mlp_fitting[-1], 3, 1)
        init.xavier_normal_(self.reconstruct_out_p_r.weight, gain=1.0)
        self.coor_refine = torch.nn.Sequential(self.fitting_mlp_r, self.reconstruct_out_p_r)   # the Neural Field Fuction (MLP)
        
        ######################## END PART : LAYERS #########################
    
    #def forward(self, points_sparse, sem_info, range_info):
    def forward(self, points_sparse, is_train=True):
        '''
        Input:
        - points_sparse: Patches xyz, (thisbatchsize, self.args.num_point, 3)
        '''

        thisbatchsize = points_sparse.size()[0]
        self.args.num_point = points_sparse.shape[1]
        #print(points_sparse.shape)
        
        ######### Set the uv_sampling_coors
        uv_sampling_coors = torch.ones([1]).float().cuda()
        if self.training == True : self.up_ratio = self.args.training_up_ratio
        else : self.up_ratio = self.args.testing_up_ratio
        self.over_sampling_up_ratio = int(self.up_ratio * self.args.over_sampling_scale) # over-sampling for steady training

        #if self.args.if_fix_sample == True: uv_sampling_coors = fix_sample(thisbatchsize, self.args.num_point, self.over_sampling_up_ratio)
        if self.args.if_fix_sample == 0: uv_sampling_coors = uniform_random_sample(thisbatchsize, self.args.num_point, self.over_sampling_up_ratio)
        elif self.args.if_fix_sample == 1: uv_sampling_coors = fix_sample(thisbatchsize, self.args.num_point, self.over_sampling_up_ratio)
        else: 
            uv_sampling_coors_1 = uniform_random_sample(thisbatchsize, self.args.num_point, self.over_sampling_up_ratio-4)
            uv_sampling_coors_2 = fix_sample(thisbatchsize, self.args.num_point, 4)
            uv_sampling_coors_ = torch.cat((uv_sampling_coors_1, uv_sampling_coors_2), dim=2) 
            uv_sampling_coors = copy.deepcopy(uv_sampling_coors_.detach())
        uv_sampling_coors = uv_sampling_coors.detach().contiguous()   # thisbatchsize, self.args.num_point, self.over_sampling_up_ratio, 2
        ######### Set the uv_sampling_coors, Done.

        neighbour_indexes_ = torch_tensor_functions.get_neighbor_index(points_sparse, self.args.feature_unfolding_nei_num)   # thisbatchsize, self.args.num_point, neighbor_num
        ######### The point-wise feature extraction, DGCNN
        # compute the point-wise feature, updated with local pooling
        neighbour_indexes_feature_extract = torch_tensor_functions.get_neighbor_index(points_sparse, self.args.neighbor_k)   # bs, vertice_num, neighbor_num
        points_in_local_patch_form = torch_tensor_functions.indexing_by_id(points_sparse,neighbour_indexes_feature_extract)
        points_in_local_patch_form = points_in_local_patch_form - points_sparse.view(thisbatchsize,self.args.num_point,1,3) # decentralized
        points_in_local_patch_form = points_in_local_patch_form.view(thisbatchsize*self.args.num_point, self.args.neighbor_k, 3)
        
        sparse_embedding = self.emb_nn_sparse(points_in_local_patch_form.transpose(1,2))  # B*num_point, self.emb_dims, self.neighbor_k
        sparse_embedding = torch.max(sparse_embedding,dim=-1,keepdim=False)[0].view(thisbatchsize,self.args.num_point,-1).permute(0,2,1)
        local_features_pooling = torch_tensor_functions.indexing_neighbor(sparse_embedding.transpose(1,2), neighbour_indexes_).permute(0,3,2,1)
        local_features_pooling = torch.max(local_features_pooling, dim=2, keepdim=False)[0]
        sparse_embedding = torch.cat((sparse_embedding,local_features_pooling),dim=1)
        sparse_embedding = sparse_embedding.permute(0,2,1)  # thisbatchsize, self.args.num_point, self.emb_dims*2
        ######### The point-wise feature extraction, Done.

        ######### TO-DO: Feature processing(point-former)
        #Final_fea = self.Final_feature(sparse_embedding, points_sparse, sem_info, range_info)  # thisbatchsize, self.emb_dims*2, self.args.num_point
        Final_fea = self.Final_feature(sparse_embedding, points_sparse)  # thisbatchsize, self.emb_dims*2, self.args.num_point
        ######### Feature processing(point-former), Done.

        ######### Instance Embedding, MLP.
        ins_embed = self.ins_head(Final_fea).permute(0, 2, 1) # thisbatchsize, self.args.num_point, self.emb_dims*2
        ######### Instance Embedding, MLP, Done.

        ######### The Neural Field, MLP.
        #sparse_embedding = torch.cat((sparse_embedding.permute(0, 2, 1), Final_fea), dim=1)
        #sparse_embedding = self.sr_head(sparse_embedding).permute(0, 2, 1)
        # get the uv_sampling_coors_id_in_sparse
        uv_sampling_coors_id_in_sparse = torch.arange(self.args.num_point).view(1, -1, 1).long()
        uv_sampling_coors_id_in_sparse = uv_sampling_coors_id_in_sparse.expand(thisbatchsize,-1,self.over_sampling_up_ratio).reshape(thisbatchsize,-1,1)
        upsampled_p = self.convert_uv_to_xyz(uv_sampling_coors.reshape(thisbatchsize,-1,2), \
                                             uv_sampling_coors_id_in_sparse, \
                                             sparse_embedding, \
                                             points_sparse,
                                             ) # thisbatchsize, self.args.num_point*self.over_sampling_up_ratio, 3
        ######### The Neural Field, MLP, Done.

        """
        ######### Offset refinement
        refine_feature = self.sr_head(Final_fea).permute(0, 2, 1)
        uv_sampling_coors_id_in_sparse_r = torch.arange(self.args.num_point).view(1, -1, 1).long()
        uv_sampling_coors_id_in_sparse_r = uv_sampling_coors_id_in_sparse_r.expand(thisbatchsize,-1,self.over_sampling_up_ratio).reshape(thisbatchsize,-1,1)
        refine_offset = self.refine_step(uv_sampling_coors.reshape(thisbatchsize,-1,2), \
                                         uv_sampling_coors_id_in_sparse_r, \
                                         refine_feature, \
                                         points_sparse,
                                         ) # thisbatchsize, self.args.num_point*self.over_sampling_up_ratio, 3
        final_upsampled_p = upsampled_p + refine_offset
        ######### Offset refinement, Done

        if not is_train:
            upsampled_p_fps_id = pn2_utils.furthest_point_sample(final_upsampled_p.contiguous(), self.up_ratio*self.args.num_point)
            querying_points_3d = pn2_utils.gather_operation(final_upsampled_p.permute(0, 2, 1).contiguous(), upsampled_p_fps_id)
            querying_points_3d = querying_points_3d.permute(0,2,1).contiguous()
            return querying_points_3d
        return final_upsampled_p, upsampled_p, ins_embed
        """
        if not is_train:
            upsampled_p_fps_id = pn2_utils.furthest_point_sample(upsampled_p.contiguous(), self.up_ratio*self.args.num_point)
            querying_points_3d = pn2_utils.gather_operation(upsampled_p.permute(0, 2, 1).contiguous(), upsampled_p_fps_id)
            querying_points_3d = querying_points_3d.permute(0,2,1).contiguous()
            return querying_points_3d

        return upsampled_p, ins_embed
        

    def convert_uv_to_xyz(self, uv_coor, uv_coor_idx_in_sparse, sparse_embedding, points_sparse):
        # uv_coor                | should be in size : thisbatchsize, All2dQueryPointNum, 2
        # uv_coor_idx_in_sparse  | should be in size : thisbatchsize, All2dQueryPointNum, 1
        # sparse_embedding       | should be in size : thisbatchsize, sparse_point_num, embedding_dim
        # points_sparse          | should be in size : thisbatchsize, sparse_point_num, 3
        thisbatchsize = uv_coor.size()[0]
        All2dQueryPointNum = uv_coor.size()[1]
        coding_dim = 4*self.args.pe_out_L + 2
        uv_encoded = position_encoding(uv_coor.reshape(-1,2).contiguous(), self.args.pe_out_L).view(thisbatchsize, All2dQueryPointNum, coding_dim).permute(0,2,1) # bs, coding_dim, All2dQueryPointNum
        indexed_sparse_feature = torch_tensor_functions.indexing_by_id(sparse_embedding, uv_coor_idx_in_sparse)  # bs, All2dQueryPointNum, 1, embedding_num 
        indexed_sparse_feature = indexed_sparse_feature.view(thisbatchsize, All2dQueryPointNum, -1).transpose(2,1)  # bs, embedding_num, All2dQueryPointNum

        coding_with_feature = torch.cat((indexed_sparse_feature, uv_encoded), dim=1)
        out_p = self.convert_feature_to_point_2to3(coding_with_feature).view(thisbatchsize, -1, All2dQueryPointNum).permute(0,2,1)
        indexed_center_points = torch_tensor_functions.indexing_by_id(points_sparse, uv_coor_idx_in_sparse).view(thisbatchsize, All2dQueryPointNum, 3)
        out_p = out_p + indexed_center_points
        return out_p
    
    def refine_step(self, uv_coor, uv_coor_idx_in_sparse, sparse_embedding, points_sparse):
        # uv_coor                | should be in size : thisbatchsize, All2dQueryPointNum, 2
        # uv_coor_idx_in_sparse  | should be in size : thisbatchsize, All2dQueryPointNum, 1
        # sparse_embedding       | should be in size : thisbatchsize, sparse_point_num, embedding_dim
        # points_sparse          | should be in size : thisbatchsize, sparse_point_num, 3
        thisbatchsize = uv_coor.size()[0]
        All2dQueryPointNum = uv_coor.size()[1]
        coding_dim = 4*self.args.pe_out_L + 2
        uv_encoded = position_encoding(uv_coor.reshape(-1,2).contiguous(), self.args.pe_out_L).view(thisbatchsize, All2dQueryPointNum, coding_dim).permute(0,2,1) # bs, coding_dim, All2dQueryPointNum
        indexed_sparse_feature = torch_tensor_functions.indexing_by_id(sparse_embedding, uv_coor_idx_in_sparse)  # bs, All2dQueryPointNum, 1, embedding_num 
        indexed_sparse_feature = indexed_sparse_feature.view(thisbatchsize, All2dQueryPointNum, -1).transpose(2,1)  # bs, embedding_num, All2dQueryPointNum

        coding_with_feature = torch.cat((indexed_sparse_feature, uv_encoded), dim=1)
        out_p = self.coor_refine(coding_with_feature).view(thisbatchsize, -1, All2dQueryPointNum).permute(0,2,1)
        indexed_center_points = torch_tensor_functions.indexing_by_id(points_sparse, uv_coor_idx_in_sparse).view(thisbatchsize, All2dQueryPointNum, 3)
        out_p = out_p + indexed_center_points
        return out_p


#### Convert a string to num_list ####      
def convert_str_2_list(str_):
    words = str_.split(' ')
    trt = [int(x) for x in words]
    return trt
#### Compute the position code for uv or xyz. ####
def position_encoding(input_uv, pe_out_L):
    ## The input_uv should be with shape (-1, X)
    ## The returned tensor should be with shape (-1, X+2*X*L)
    ## X = 2/3 if the input is uv/xyz.
    trt = input_uv
    for i in range(pe_out_L):
        trt = torch.cat((trt, torch.sin(input_uv*(2**i)*(3.14159265))) , dim=-1 )
        trt = torch.cat((trt, torch.cos(input_uv*(2**i)*(3.14159265))) , dim=-1 )
    return trt
#### Sample uv by a fixed manner. #### 
def fix_sample(thisbatchsize, num_point, up_ratio, if_random=False):
    if if_random==True: 
        print('Random sampling mode is not supported right now.')
        exit()
    if up_ratio == 4:
        one_point_fixed = [ [ [0,0] for i in range(2)] for j in range(2) ]
        for i in range(2):
            for j in range(2):
                one_point_fixed[i][j][0] = (i/1) *2 -1
                one_point_fixed[i][j][1] = (j/1) *2 -1
        one_point_fixed = np.array(one_point_fixed).reshape(4,2)
        one_batch_uv2d_random_fixed = np.expand_dims(one_point_fixed,axis=0)
        one_batch_uv2d_random_fixed = np.expand_dims(one_batch_uv2d_random_fixed,axis=0)
        one_batch_uv2d_random_fixed = np.tile(one_batch_uv2d_random_fixed,[thisbatchsize, num_point, 1,1])
        one_batch_uv2d_random_fixed_tensor = torch.from_numpy(one_batch_uv2d_random_fixed).cuda().float()
        return one_batch_uv2d_random_fixed_tensor
    if up_ratio == 9:
        one_point_fixed = [ [ [0,0] for i in range(3)] for j in range(3) ]
        for i in range(3):
            for j in range(3):
                one_point_fixed[i][j][0] = (i/2) *2 -1
                one_point_fixed[i][j][1] = (j/2) *2 -1
        one_point_fixed = np.array(one_point_fixed).reshape(9,2)
        one_batch_uv2d_random_fixed = np.expand_dims(one_point_fixed,axis=0)
        one_batch_uv2d_random_fixed = np.expand_dims(one_batch_uv2d_random_fixed,axis=0)
        one_batch_uv2d_random_fixed = np.tile(one_batch_uv2d_random_fixed,[thisbatchsize, num_point, 1,1])
        one_batch_uv2d_random_fixed_tensor = torch.from_numpy(one_batch_uv2d_random_fixed).cuda().float()
        return one_batch_uv2d_random_fixed_tensor
    if up_ratio == 16:
        one_point_fixed = [ [ [0,0] for i in range(4)] for j in range(4) ]
        for i in range(4):
            for j in range(4):
                one_point_fixed[i][j][0] = (i/3) *2 -1
                one_point_fixed[i][j][1] = (j/3) *2 -1
        one_point_fixed = np.array(one_point_fixed).reshape(16,2)
        one_batch_uv2d_random_fixed = np.expand_dims(one_point_fixed,axis=0)
        one_batch_uv2d_random_fixed = np.expand_dims(one_batch_uv2d_random_fixed,axis=0)
        one_batch_uv2d_random_fixed = np.tile(one_batch_uv2d_random_fixed,[thisbatchsize, num_point, 1,1])
        one_batch_uv2d_random_fixed_tensor = torch.from_numpy(one_batch_uv2d_random_fixed).cuda().float()
        return one_batch_uv2d_random_fixed_tensor
    if up_ratio == 64:
        one_point_fixed = [ [ [0,0] for i in range(8)] for j in range(8) ]
        for i in range(8):
            for j in range(8):
                one_point_fixed[i][j][0] = (i/7) *2 -1
                one_point_fixed[i][j][1] = (j/7) *2 -1
        one_point_fixed = np.array(one_point_fixed).reshape(64,2)
        one_batch_uv2d_random_fixed = np.expand_dims(one_point_fixed,axis=0)
        one_batch_uv2d_random_fixed = np.expand_dims(one_batch_uv2d_random_fixed,axis=0)
        one_batch_uv2d_random_fixed = np.tile(one_batch_uv2d_random_fixed,[thisbatchsize, num_point, 1,1])
        one_batch_uv2d_random_fixed_tensor = torch.from_numpy(one_batch_uv2d_random_fixed).cuda().float()
        return one_batch_uv2d_random_fixed_tensor
    if up_ratio == 1024:
        one_point_fixed = [ [ [0,0] for i in range(32)] for j in range(32) ]
        for i in range(32):
            for j in range(32):
                one_point_fixed[i][j][0] = (i/31) *2 -1
                one_point_fixed[i][j][1] = (j/31) *2 -1
        one_point_fixed = np.array(one_point_fixed).reshape(1024,2)
        one_batch_uv2d_random_fixed = np.expand_dims(one_point_fixed,axis=0)
        one_batch_uv2d_random_fixed = np.expand_dims(one_batch_uv2d_random_fixed,axis=0)
        one_batch_uv2d_random_fixed = np.tile(one_batch_uv2d_random_fixed,[thisbatchsize, num_point, 1,1])
        one_batch_uv2d_random_fixed_tensor = torch.from_numpy(one_batch_uv2d_random_fixed).cuda().float()
        return one_batch_uv2d_random_fixed_tensor
    else:
        print('This up_ratio ('+str(up_ratio)+') is not supported now. You can try the random mode!')
        exit()
#### Sample uv uniformly in (-1,1). #### 
def uniform_random_sample(thisbatchsize, num_point, up_ratio):
    # return : randomly and uniformly sampled uv_coors   |   Its shape should be : thisbatchsize, num_point, up_ratio, 2
    res_ = torch.rand(thisbatchsize*num_point, 4*up_ratio, 3)*2-1
    res_ = res_.cuda()
    res_[:,:,2:]*=0
    furthest_point_index = pn2_utils.furthest_point_sample(res_,up_ratio)
    uniform_res_ = pn2_utils.gather_operation(res_.permute(0, 2, 1).contiguous(), furthest_point_index)
    uniform_res_ = uniform_res_.permute(0,2,1).contiguous()
    uniform_res_ = uniform_res_[:,:,:2].view(thisbatchsize, num_point, up_ratio, 2)
    return uniform_res_
#### Compute the grad ####
def cal_grad(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad = False, device = outputs.device)
    points_grad = grad(
        outputs = outputs,
        inputs = inputs,
        grad_outputs = d_points,
        create_graph = True,
        retain_graph = True,
        only_inputs = True)[0]
    return points_grad
######## TODO: END PART: OUR OWN NETWORK ########
