from turtle import forward
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class TransformerEncoderLayerPreNorm(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        '''
        d_model: input and output dim
        dim_feedforward: hidden dim
        '''
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, src, src_mask = None, src_key_padding_mask = None):
        # return [N, B, C]
        src = self.norm1(src)
        src2, mask = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        
        # coordinate refinement
        src = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        
        return src

class TransformerDecoderLayerPreNorm(nn.Module):

    def __init__(self, d_model, nc_mem, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm_mem = nn.LayerNorm(nc_mem)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        
        tgt = self.norm1(tgt)
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)

        tgt = self.norm2(tgt)
        memory = self.norm_mem(memory)
        tgt2, mask = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        
        tgt = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        
        return tgt


class ConvModule(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, padding=0):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(c_out)
        self.act = nn.ReLU()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

######## TODO: START PART: Semantic label process, MLP with ReLU. ########
class Feature_MLPNet_relu(torch.nn.Module):
    """ Multi-layer perception.
        in_channel = the semantic label number
        [B, Cin, N] -> [B, Cout, N]
    """
    def __init__(self, in_channel, out_channel=None, hidden_dim=None, bn_momentum=0.1):
        super().__init__()
        if out_channel is None:
            out_channel = in_channel
        if hidden_dim is None:
            hidden_dim = int(in_channel/4)
        self.layers = torch.nn.Sequential(
            torch.nn.Conv1d(in_channel, hidden_dim, 1),
            torch.nn.BatchNorm1d(hidden_dim, momentum=bn_momentum), 
            torch.nn.ReLU(),
            torch.nn.Conv1d(hidden_dim, hidden_dim, 1),
            torch.nn.BatchNorm1d(hidden_dim, momentum=bn_momentum), 
            torch.nn.ReLU(),
            torch.nn.Conv1d(hidden_dim, out_channel, 1),
        )
        self.layers.apply(self.init_weight)
    def forward(self, inp):
        #print(inp.shape)
        out = self.layers(inp)
        return out
    def init_weight(self, module):
        # custom weights initialization
        if isinstance(module, nn.Linear):
            init.xavier_normal_(module.weight, gain=1.0)
######## TODO: END PART: Semantic label process, MLP with ReLU. ########

class LocalTransformer(nn.Module):
    def __init__(self, dim_feature, dim_out, fea_in_dim, nhead=4, num_layers=2, drop=0.0, prenorm=True):
        super().__init__()
        self.nc_in = dim_feature # transformer block channel input dim
        self.nc_out = dim_out # transformer block channel output dim

        # Feature Processing mlps
        self.Geo_mlp = Feature_MLPNet_relu(fea_in_dim, self.nc_in)
        # Position Encoding conv.
        self.pe = nn.Sequential(
            ConvModule(3, self.nc_in // 2, 1),
            ConvModule(self.nc_in // 2, self.nc_in, 1)
        )
        self.he = nn.Sequential(
            ConvModule(1, self.nc_in // 2, 1),
            ConvModule(self.nc_in // 2, self.nc_in, 1)
        )
        self.initial_processing = Feature_MLPNet_relu(3*self.nc_in, self.nc_in)

        # point-former blocks
        BSC_Encoder = TransformerEncoderLayerPreNorm if prenorm else nn.TransformerEncoderLayer
        self.chunk = nn.TransformerEncoder(
            BSC_Encoder(d_model=self.nc_in, dim_feedforward=2*self.nc_in, dropout=drop, nhead=nhead),
            num_layers=num_layers)
        # final output layer
        self.fc = ConvModule(self.nc_in, self.nc_out, 1)

    #def forward(self, Geo_fea, xyz, Sem_info, Range_info):
    def forward(self, Geo_fea, xyz):
        '''
        Input:
        - Geo_fea: Local geometric feature, [B, N, dim_feature]
        - xyz: Position Encoding, [B, N, 3]
        '''
        
        min_z = torch.quantile(xyz[:, :, 2], q=0.1, dim=-1)
        relative_z = xyz[:, :, 2:3] - min_z

        ## Init Feature Processing
        Geo_fea_init = self.Geo_mlp(Geo_fea.permute(0, 2, 1)) # bs, nc_in, N
        Pos_En = self.pe(xyz.permute(0, 2, 1)) # bs, nc_in, N
        Hei_En = self.he(relative_z.permute(0, 2, 1)) # bs, nc_in, N

        Init_fea = torch.cat((Geo_fea_init, Pos_En, Hei_En), dim=1)
        Init_fea = self.initial_processing(Init_fea).permute(2, 0, 1) # N, bs, nc_in

        Atten_fea = self.chunk(Init_fea).permute(1, 2, 0) # (B, C, npoint)
        #print('Atten_fea: ', Atten_fea.shape)
        output_features = self.fc(Atten_fea)

        #input_features = input_features.permute(0, 2, 1, 3).reshape(-1, D, ns).permute(2, 0, 1) # [ns, B*np, D]
        #transformed_feats = self.chunk(input_features).permute(1, 2, 0).reshape(B, np, D, ns).transpose(1, 2)
        #output_features = F.max_pool2d(transformed_feats, kernel_size=[1, ns]) # (B, C, npoint)
        #output_features = self.fc(output_features).squeeze(-1)

        return output_features


class LocalGlobalTransformer(nn.Module):

    #def __init__(self, dim_in, dim_out, nhead=4, num_layers=2, norm_cfg=dict(type='BN2d'), ratio=1, mem_pts=20000, tgt_pts=2048, drop=0.0, dim_feature=64, prenorm=True):
    def __init__(self, dim_in, dim_out, nhead=4, num_layers=2, drop=0.0, dim_feature=64, prenorm=True):
        super().__init__()
        self.nc_in = dim_in
        self.nc_out = dim_out
        self.nhead = nhead
        self.global_fea_process = ConvModule(32, self.nc_in, 1)
        self.pe = nn.Sequential(
            ConvModule(3, self.nc_in // 2, 1),
            ConvModule(self.nc_in // 2, self.nc_in, 1)
            )

        BSC_Decoder = TransformerDecoderLayerPreNorm if prenorm else nn.TransformerDecoderLayer
        
        self.chunk = nn.TransformerDecoder(
            BSC_Decoder(d_model=self.nc_in, dim_feedforward=2*self.nc_in, dropout=drop, nhead=nhead, nc_mem=dim_feature),
            num_layers=num_layers)
        
        self.fc = ConvModule(self.nc_in, self.nc_out, 1)
        

    def forward(self, xyz_tgt, xyz_mem, features_tgt, features_mem):
        '''
        - xyz_tgt: local xyz, [B, N, 3]
        - xyz_mem: global xyz, [B, N, 3]
        - features_tgt: local feature, [B, C, N]
        - features_mem: global feature, [B, C, N]
        '''
        #xyz_tgt_flipped = xyz_tgt.transpose(1, 2)
        #xyz_mem_flipped = xyz_mem.transpose(1, 2)
        
        #tgt = features_tgt.unsqueeze(-1) + self.pe(xyz_tgt_flipped)
        #mem = features_mem.unsqueeze(-1) + self.pe(xyz_mem_flipped)

        tgt = features_tgt + self.pe(xyz_tgt.transpose(1, 2))
        features_mem = features_mem.permute(0, 2, 1)
        features_mem = self.global_fea_process(features_mem)
        mem = features_mem + self.pe(xyz_mem.transpose(1, 2))

        mem_mask = None
        
        mem = mem.permute(2, 0, 1) # (N, B, C)
        tgt = tgt.permute(2, 0, 1) # (N, B, C)
        #print(mem.shape)
        #print(tgt.shape)

        transformed_feats = self.chunk(tgt, mem, memory_mask=mem_mask).permute(1, 2, 0) # (B, C, N)
        output_features = self.fc(transformed_feats)
        
        return output_features


