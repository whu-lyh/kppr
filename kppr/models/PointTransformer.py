
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils
from timm.layers import DropPath, trunc_normal_

# Modified from Point-BERT backbone

def fps(data, number):
    '''
        pointnet cuda based Farest points sampling
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data


def square_distance(src, dst):
    """
        Calculate Euclid distance between each two points.
        src^T * dst = xn * xm + yn * ym + zn * zm；
        sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
        sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
        dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
            = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
        Input:
            src: source points, [B, N, C]
            dst: target points, [B, M, C]
        Output:
            dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


# ref https://github.com/Strawberry-Eat-Mango/PCT_Pytorch/blob/main/util.py
def knn_point(nsample, xyz, new_xyz):
    """
        Input:
            nsample: max sample number in local region
            xyz: all points, [B, N, C]
            new_xyz: query points, [B, S, C]
        Return:
            group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx


class Group(nn.Module):
    '''
        Divide the input point cloud into groups
        Input:
            points
        Return:
            the neighborbood points and normalized centorid
    '''
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        #self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out, nearly randomly selected
        center = fps(xyz, self.num_group) # B G 3
        # knn to get the neighborhood using cuda acceleration
        # _, idx = self.knn(xyz, center) # B G M
        # or without cuda
        idx = knn_point(self.group_size, xyz, center) # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


class MiniPointNetEncoder(nn.Module):
    '''
        Mini-PointNet style encoder for point embeddings
    '''
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )
        
    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        '''
            Input: batch_size, number of patches(also with position embedding), token embedding dimension
            basically following the mathmetic formula
        '''
        B, N, C = x.shape
        #qkv->[batchsize, num_patches+1, 3*total_embed_dim]
        #reshape->[batchsize, num_patches+1, 3, num_heads, embed_dim_per_head]
        #permute->[3, batchsize, num_heads, num_patches+1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        #q  [batchsize, num_heads, num_patches+1, embed_dim_per_head]
        #k^T[batchsize, num_heads, embed_dim_per_head, num_patches+1]
        #q*k^T=[batchsize, num_heads, num_patches+1, num_patches+1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # softmax following the last dimension
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        #@->[batchsize, num_heads, num_patches+1, embed_dim_per_head]
        #transpose->[batchsize, num_patches+1, num_heads, embed_dim_per_head]
        #reshape->[batchsize, num_patches+1, num_heads*embed_dim_per_head]即[batchsize, num_patches+1, total_embed_dim]
        #reshape infact equals to concatenation operation
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """ 
        Transformer Encoder without hierarchical structure, specifically no patch embedding layer
    """
    def __init__(self, embed_dim=768, depth=4, num_heads=12, 
                mlp_ratio=4., qkv_bias=False, qk_scale=None,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
                )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x


class PointTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        # basic transformer building parameters
        self.config = config
        # token embedding dimension
        self.trans_dim = 384 # config.trans_dim
        # transformer block number
        self.depth = 12# config.depth 
        self.drop_path_rate = 0.1 # config.drop_path_rate 
        self.num_heads = 12 # config.num_heads
        # global_pool: Type of global pooling for final sequence (default: 'token')
        global_pool = 'avg' # config.global_pool
        # class_token: Use class token
        class_token = False # config.class_token
        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'
        self.num_classes = 1000 # config.num_classes
        self.global_pool = 'avg' # config.global_pool
        self.num_prefix_tokens = 1 if class_token else 0
        fc_norm = True # config.fc_norm
        # fc_norm: Pre head norm after pool (instead of before), if None, enabled when global_pool == 'avg'
        use_fc_norm = self.global_pool == 'avg' if fc_norm is None else fc_norm
        # raw point cloud to group center with neighborhood points
        # neighborhood point number
        self.group_size = 32 # config.group_size 
        # number of center
        self.num_group = 2000 # config.num_group
        # grouper
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        # define the patch embedding encoder
        # patch embedding feature dimension, emb_encoder_dim is the feature dimension from patch embedding layer
        self.emb_encoder_dim = 256 # config.emb_encoder_dim 
        self.emb_encoder = MiniPointNetEncoder(encoder_channel=self.emb_encoder_dim)
        # bridge encoder and transformer
        self.reduce_dim = nn.Linear(self.emb_encoder_dim, self.trans_dim)
        # learnable position embedding, 3-128-self.trans_dim
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        # FIXME: alternative manner?
        #self.pos_embed = nn.Parameter(torch.randn(1, embed_len, trans_dim) * .02)
        # set transformer blocks
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads
        )
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(self.trans_dim) if use_fc_norm else nn.Identity()
        # the heads are already concated inside attention blocks
        # Global Average Pooling
        self.GAP = nn.AdaptiveAvgPool1d(1)
        # if gap, then comment the following classifier head
        # Classifier Head
        # self.fc_norm = norm_layer(self.trans_dim) if use_fc_norm else nn.Identity()
        # self.head_drop = nn.Dropout(0.)
        # self.head = nn.Linear(self.trans_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        # FIXME: bug related to pos embedding manner
        #trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
      
    def reset_classifier(self, num_classes: int, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token')
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity() 

    def load_model_from_ckpt(self, bert_ckpt_path):
        ckpt = torch.load(bert_ckpt_path)
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        for k in list(base_ckpt.keys()):
            if k.startswith('transformer_q') and not k.startswith('transformer_q.cls_head'):
                base_ckpt[k[len('transformer_q.'):]] = base_ckpt[k]
            elif k.startswith('base_model'):
                base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
            del base_ckpt[k]

        incompatible = self.load_state_dict(base_ckpt, strict=False)

        if incompatible.missing_keys:
            print('missing_keys')
            print(get_missing_parameters_message(incompatible.missing_keys))
        if incompatible.unexpected_keys:
            print('unexpected_keys')
            print(get_unexpected_parameters_message(incompatible.unexpected_keys))

        print(f'[PointTransformer] Successful Loading the ckpt from {bert_ckpt_path}')

    def forward_features(self, pts):
        # divide the point cloud in the same form. This is important
        neighborhood, center = self.group_divider(pts)
        # encode the point cloud patches
        group_input_tokens = self.emb_encoder(neighborhood) # B G N1
        # bridge the patch embedding into target dimension
        group_input_tokens = self.reduce_dim(group_input_tokens) # B G N2
        # add pos embedding based on the center pts
        pos = self.pos_embed(center) # B G N2
        # feed final input into transformer blocks
        x = self.blocks(group_input_tokens, pos)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool=False):
        if self.global_pool:
            #if global_pool is set using the pooling vector as the final classification feature instead of cls token
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward_global_feat(self, x, feat_type:str=''):
        if feat_type == 'GAP':
            x = self.GAP(x.transpose(-1, -2)).squeeze(2)
        else:
            NotImplementedError()
        return x

    def forward(self, x):
        x = self.forward_features(x)
        # FIXME What's usage of this kind of head?
        # x = self.forward_head(x)
        x = self.forward_global_feat(x, feat_type='GAP')
        return x