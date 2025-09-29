import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import ot
import numpy as np
from .MMD import batch_mmd
from .components_mmp import  MMAttentionLayer, FeedForward, FeedForwardEnsemble, process_surv 


def agg_histo(X, agg_mode='mean'):
    """
    Aggregating histology
    """
    if agg_mode == 'mean':
        out = torch.mean(X, dim=1)
    elif agg_mode == 'cat':
        out = X.reshape(X.shape[0], -1)
    else:
        raise NotImplementedError(f"Not implemented for {agg_mode}")

    return out


class OT_Attn(nn.Module):
    """
    Optimal transport
    """
    def __init__(self, impl='pot-uot-l2', ot_reg=0.1, ot_tau=0.5) -> None:
        super().__init__()
        self.impl = impl
        self.ot_reg = ot_reg
        self.ot_tau = ot_tau
        print("ot impl: ", impl)

    def normalize_feature(self, x):
        x = x - x.min(-1)[0].unsqueeze(-1)
        return x

    def OT(self, weight1, weight2):
        """
        Parmas:
            weight1 : (B, N, D)
            weight2 : (B, M, D)

        Return:
            flow : (N, M)
            dist : (1, )
        """

        if self.impl == "pot-uot-l2":
            a, b = torch.tensor(ot.unif(weight1.size()[0])).cuda(), torch.tensor(ot.unif(weight2.size()[0])).cuda()
            self.cost_map = torch.cdist(weight1, weight2) ** 2 

            cost_map_detach = self.cost_map
            M_cost = cost_map_detach / cost_map_detach.max()

            flow = ot.unbalanced.sinkhorn_knopp_unbalanced(a=a, b=b,
                                                           M=M_cost.double(), reg=self.ot_reg,
                                                           reg_m=self.ot_tau)

            dist = self.cost_map * flow  
            return flow
        else:
            raise NotImplementedError(f"Not implemented for {self.impl}!")

    def forward(self, x, y):
        '''
        x: (N, 1, D)
        y: (M, 1, D)
        '''

        pi = [self.OT(self.normalize_feature(x[idx]), self.normalize_feature(y[idx])) for idx in range(len(x))]
        pi = torch.stack(pi, dim=0)

        return pi.transpose(2, 1)


def construct_proto_embedding(path_proj_dim, append_embed='modality', numOfproto_histo=16, numOfproto_omics=50):
    """
    Per-prototype learnable/non-learnable embeddings to append to the original prototype embeddings 
    """
    if append_embed == 'modality':  
        path_proj_dim_new = path_proj_dim + 2

        histo_embedding = torch.tensor([[[1, 0]]]).repeat(1, numOfproto_histo, 1)  
        gene_embedding = torch.tensor([[[0, 1]]]).repeat(1, numOfproto_omics, 1) 

    elif append_embed == 'proto':
        path_proj_dim_new = path_proj_dim + numOfproto_histo + numOfproto_omics
        embedding = torch.eye(numOfproto_histo + numOfproto_omics).unsqueeze(0)

        histo_embedding = embedding[:, :numOfproto_histo, :]  
        gene_embedding = embedding[:, numOfproto_histo:, :]  

    elif append_embed == 'random':
        append_dim = 32
        path_proj_dim_new = path_proj_dim + append_dim

        histo_embedding = torch.nn.Parameter(torch.randn(1, numOfproto_histo, append_dim), requires_grad=True)
        gene_embedding = torch.nn.Parameter(torch.randn(1, numOfproto_omics, append_dim), requires_grad=True)

    else:
        path_proj_dim_new = path_proj_dim
        histo_embedding = None
        gene_embedding = None

    return path_proj_dim_new, histo_embedding, gene_embedding



################################
# Multimodal fusion approaches #
################################
class coattn(nn.Module):
    def __init__(
            self,
            omic_sizes=[100, 200, 300, 400, 500, 600],
            histo_in_dim=768,
            dropout=0.1,
            num_classes=4,
            path_proj_dim=256,
            num_coattn_layers=1,
            modality='both',
            histo_agg='mean',
            histo_model='PANTHER',
            append_embed='none',
            mult=1,
            net_indiv=False,
            numOfproto=16,
            type_layer="Mamba",
            lr_alpha=0.05,
            lr_beta=0.05,
            text_in_dim=768):
        """
        The central co-attention module where you can do it all!

        Args:
            text_sizes: List of integers, each indicating number of genes per prototype
            histo_in_dim: Dimension of histology feature embedding
            num_classes: 4 if we are using NLL, 1 if we are using Cox/Ranking loss
            path_proj_dim: Dimension of the embedding space where histology and pathways are fused
            modality: ['gene','histo','coattn', 'partial'] 'coattn' accounts for both modalities
                If 'histo' or 'gene', unimodal self-attention
            histo_agg: ['mean', 'cat'] Take average of post-attention embeddings ('mean') or concatenate ('cat')
            histo_model: ['mil','PANTHER', 'OT', 'H2T']: 'mil' is for non-prototype-based methods
            net_indiv (bool): If True, create FFN for each prototype
            numOfproto: Number of histology prototypes
        """

        super().__init__()


        self.num_text_proto= 28
        self.num_coattn_layers = num_coattn_layers

        self.histo_in_dim = histo_in_dim
        self.text_in_dim = text_in_dim
        self.out_mult = mult
        self.net_indiv = net_indiv
        self.modality = modality

        self.histo_agg = histo_agg

        self.numOfproto = numOfproto
        self.num_classes = num_classes

        self.histo_model = histo_model.lower()

        self.identity = nn.Identity()  

        self.append_embed = append_embed
        
        if self.histo_model == 'panther':  
            self.path_proj_net = nn.Sequential(nn.Linear(self.histo_in_dim * 2 + 1, path_proj_dim))
        else:
            self.path_proj_net = nn.Sequential(nn.Linear(self.histo_in_dim, path_proj_dim))
            
        self.text_proj_net = nn.Sequential(nn.Linear(self.text_in_dim, path_proj_dim))

        self.diff_path_proj_net = nn.Sequential(nn.Linear(self.histo_in_dim, path_proj_dim))
        self.diff_text_proj_net = nn.Sequential(nn.Linear(self.text_in_dim, path_proj_dim))



        #
        if self.histo_model != "mil":
            self.path_proj_dim, self.histo_embedding, self.gene_embedding = construct_proto_embedding(path_proj_dim,
                                                                                                      self.append_embed,
                                                                                                      self.numOfproto,
                                                                                                      self.num_text_proto)
        else:
            self.path_proj_dim = path_proj_dim
            self.histo_embedding = None
            self.gene_embedding = None

 
        coattn_list = []
        if self.num_coattn_layers == 0:
            out_dim = self.path_proj_dim

            if self.net_indiv:  # Individual MLP per prototype
                feed_forward = FeedForwardEnsemble(out_dim,
                                                   self.out_mult,
                                                   dropout=dropout,
                                                   num=self.numOfproto + self.num_text_proto)
            else:
                feed_forward = FeedForward(out_dim, self.out_mult, dropout=dropout)

            layer_norm = nn.LayerNorm(int(out_dim * self.out_mult))
            coattn_list.extend([feed_forward, layer_norm])
        else:
            out_dim = self.path_proj_dim     
            out_mult = self.out_mult
            
            if self.modality in ['histo', 'gene']: 
                attn_mode = 'self'
            else:
                attn_mode = 'full'  

            cross_attender = MMAttentionLayer(
                dim=self.path_proj_dim,
                dim_head=out_dim,
                heads=1,
                residual=False,
                dropout=0.1,
                num_pathways=self.num_text_proto,
                attn_mode=attn_mode
            )

            if self.net_indiv:  
                feed_forward = FeedForwardEnsemble(out_dim,
                                                   out_mult,
                                                   dropout=dropout,
                                                   num=self.numOfproto+16 + self.num_text_proto)
            else:
                feed_forward = FeedForward(out_dim, out_mult, dropout=dropout)

            layer_norm = nn.LayerNorm(int(out_dim * out_mult))
            coattn_list.extend([cross_attender, feed_forward, layer_norm])


        self.coattn = nn.Sequential(*coattn_list)

        out_dim_final = int(out_dim * self.out_mult)
        histo_final_dim = out_dim_final * self.numOfproto if self.histo_agg == 'cat' else out_dim_final
        gene_final_dim = out_dim_final

        if self.modality == 'histo':
            in_dim = histo_final_dim
        elif self.modality == 'gene':
            in_dim = gene_final_dim
        else:
            in_dim = histo_final_dim + gene_final_dim

        self.classifier = nn.Linear(in_dim, self.num_classes, bias=False)
        from mil_models.club import MIEstimator
        self.CLUB = MIEstimator(self.path_proj_dim)

        from mil_models.mamba_agg import MM_SS2D
        self.AAM_path = MM_SS2D(d_model=self.path_proj_dim,type_layer=type_layer)
        self.AAM_text = MM_SS2D(d_model=self.path_proj_dim,type_layer=type_layer)

        self.lr_alpha = lr_alpha
        self.lr_beta = lr_beta
        
    def forward_no_loss(self, x_path, x_text, x_diff_path, x_diff_text, return_attn=False):
        """
        Args:
            x_path: (B, numOfproto, in_dim) in_dim = [prob, mean, cov] (If OT, prob will be uniform, cov will be none)
            x_omics:
            return_attn:

        """
        device = x_path.device
        h_text = self.text_proj_net(x_text)
        h_path = self.path_proj_net(x_path)

        h_diff_path = self.diff_path_proj_net(x_diff_path)
        h_diff_text = self.diff_text_proj_net(x_diff_text)

        if self.histo_embedding is not None:   
            arr = []
            for idx in range(len(h_path)):
                arr.append(torch.cat([h_path[idx:idx + 1], self.histo_embedding.to(device)], dim=-1))
            h_path = torch.cat(arr, dim=0)
        
        #AAM
        h_path,h_diff_path = self.AAM_path(h_path,h_diff_path)
        h_text,h_diff_text = self.AAM_text(h_text,h_diff_text)

        tokens_p = torch.cat([h_path, h_diff_path], dim=1) 
        tokens_t = torch.cat([h_text, h_diff_text], dim=1) 
        tokens= torch.cat([tokens_t,tokens_p], dim=1) 

        if return_attn:
            with torch.no_grad():
                _, attn_pathways, cross_attn_pathways, cross_attn_histology = self.coattn[0](x=tokens, mask=None, return_attention=True)

        mm_embed = self.coattn(tokens)

        # tEXT
        paths_postSA_embed = mm_embed[:, :self.num_text_proto, :]
        paths_postSA_embed = torch.mean(paths_postSA_embed, dim=1)

        # Histology
        wsi_postSA_embed = mm_embed[:, self.num_text_proto:, :]
        
        if self.histo_model == 'mil':
            wsi_postSA_embed = torch.mean(wsi_postSA_embed, dim=1)  # For non-prototypes, we just take the mean
        else:
            wsi_postSA_embed = agg_histo(wsi_postSA_embed, self.histo_agg)

        if self.modality == 'histo':    # Just use histo for prediction
            embedding = wsi_postSA_embed
        elif self.modality == 'gene':   # Just use gene for prediction
            embedding = paths_postSA_embed
        else:   # Use both modalities
            embedding = torch.cat([paths_postSA_embed, wsi_postSA_embed], dim=1)  # ---> both branches

        logits = self.classifier(embedding)
        out = {'logits': logits}
        if return_attn:
            out['omic_attn'] = attn_pathways
            out['cross_attn'] = cross_attn_pathways
            out['path_attn'] = cross_attn_histology
        
        #ibd
        out['h_path'],out['h_diff_path']= h_path,h_diff_path
        out['h_text'],out['h_diff_text']= h_text,h_diff_text

        out['histology'], out['pathways'] =torch.mean(tokens_p, dim=1),torch.mean(tokens_t, dim=1)
        out['global_embed'] = torch.mean(torch.cat([mm_embed[:, :self.num_text_proto, :],mm_embed[:, self.num_text_proto:, :]], dim=1), dim=1)

        return out


    def forward(self, h, model_kwargs={},text_emb=None):

        attn_mask = model_kwargs['attn_mask']
        label = model_kwargs['label']
        censorship = model_kwargs['censorship']
        loss_fn = model_kwargs['loss_fn']

        x_diff_t,x_diff_p,x_gs =h[...,:768],h[...,768:768*2],h[...,768*2:]
        out = self.forward_no_loss(x_path=x_gs,x_text=text_emb,x_diff_path=x_diff_p,x_diff_text=x_diff_t)
        logits = out['logits']

        results_dict, log_dict = process_surv(logits, label, censorship, loss_fn)
        
        histology, pathways, global_embed =out['histology'], out['pathways'], out['global_embed']

        mimin = self.CLUB(histology, pathways, global_embed)
        mimin_loss = self.CLUB.learning_loss(histology, pathways, global_embed)

        mmd_p= batch_mmd( out['h_path'],out['h_diff_path']).mean()
        mmd_t= batch_mmd( out['h_text'],out['h_diff_text']).mean()

        results_dict['loss'] = results_dict['loss'] + self.lr_alpha*(mmd_p+mmd_t) + self.lr_beta*(mimin_loss+ mimin)
        

        return results_dict, log_dict



class coattn_mot(nn.Module):
    def __init__(
            self,
            omic_sizes=[100, 200, 300, 400, 500, 600],
            histo_in_dim=768,
            dropout=0.1,
            num_classes=4,
            path_proj_dim=128,
            num_coattn_layers=1,
            modality='both',
            histo_agg='mean',
            histo_model='panther',
            append_embed='none',
            mult=1,
            net_indiv=False,
            numOfproto=16,
            ot_reg=0.1,
            ot_tau=0.5,
            type_layer="Mamba",
            lr_alpha=0.05,
            text_in_dim=768,
            ot_impl="pot-uot-l2"):
        """
        Multimodal fusion with cross-modal optimal transport (instead of Transformer cross-attention).
        Visualization currently not supported

        Args:
            omic_sizes: List of integers, each indicating number of genes per prototype
            histo_in_dim: Dimension of histology feature embedding
            num_classes: 4 if we are using NLL, 1 if we are using Cox/Ranking loss
            path_proj_dim: Dimension of the embedding space where histology and pathways are fused
            modality: ['gene','histo','coattn'] 'coattn' accounts for both modalities
            histo_agg: ['mean', 'cat'] Take average of post-attention embeddings ('mean') or concatenate ('cat')
            histo_model: ['mil','PANTHER', 'OT', 'H2T']: 'mil' is for non-prototype-based methods
            net_indiv (bool): If True, create FFN for each prototype
            numOfproto: Number of histology prototypes
        """

        super().__init__()

        self.num_text_proto= 12 + 16 
        self.num_coattn_layers = num_coattn_layers

        self.histo_in_dim = histo_in_dim
        self.text_in_dim = text_in_dim
        self.out_mult = mult
        self.net_indiv = net_indiv
        self.modality = modality

        self.histo_agg = histo_agg

        self.numOfproto = numOfproto
        self.num_classes = num_classes

        self.histo_model = histo_model.lower()

        if self.histo_model == 'panther':  # Uses prob/mean/cov
            self.path_proj_net = nn.Sequential(nn.Linear(self.histo_in_dim * 2 + 1, path_proj_dim))
        else:
            self.path_proj_net = nn.Sequential(nn.Linear(self.histo_in_dim, path_proj_dim))
        self.text_proj_net = nn.Sequential(nn.Linear(self.histo_in_dim, path_proj_dim))

        self.diff_path_proj_net = nn.Sequential(nn.Linear(self.histo_in_dim, path_proj_dim))
        self.diff_text_proj_net = nn.Sequential(nn.Linear(self.text_in_dim, path_proj_dim))
        
        self.identity = nn.Identity()  # use this layer to calculate ig

        self.append_embed = append_embed

        if histo_model != "mil":
            self.path_proj_dim, self.histo_embedding, self.gene_embedding = construct_proto_embedding(path_proj_dim,
                                                                                                      self.append_embed,
                                                                                                      self.numOfproto,
                                                                                                      len(omic_sizes))
        else:
            self.path_proj_dim = path_proj_dim
            self.histo_embedding = None
            self.gene_embedding = None       


        out_dim = self.path_proj_dim   
        self.out_mult = mult
        self.mot = OT_Attn(impl=ot_impl, ot_reg=ot_reg, ot_tau=ot_tau)

        coattn = MMAttentionLayer(
                                    dim=self.path_proj_dim,
                                    dim_head=out_dim,
                                    heads=1,
                                    residual=False,
                                    dropout=0.1,
                                    num_pathways=self.num_text_proto,
                                    attn_mode='self'
                                )

        if self.net_indiv:  # Individual MLP per prototype
            feed_forward = FeedForwardEnsemble(out_dim,
                                               self.out_mult,
                                               dropout=dropout,
                                               num=self.numOfproto + self.num_text_proto +16) 
        else:
            feed_forward = FeedForward(out_dim, self.out_mult, dropout=dropout)

        layer_norm = nn.LayerNorm(int(out_dim * self.out_mult))

        self.coattn = nn.Sequential(coattn, feed_forward, layer_norm)

        out_dim_final = int(out_dim * self.out_mult)
        histo_final_dim = out_dim_final * self.numOfproto if self.histo_agg == 'cat' else out_dim_final
        gene_final_dim = out_dim_final

        if self.modality == 'histo':
            in_dim = histo_final_dim
        elif self.modality == 'gene':
            in_dim = gene_final_dim
        else:
            in_dim = histo_final_dim + gene_final_dim

        self.classifier = nn.Linear(in_dim, self.num_classes, bias=False)

        from mil_models.club import MIEstimator
        self.CLUB = MIEstimator(self.path_proj_dim)

        from mil_models.mamba_agg import MM_SS2D
        self.AAM_path = MM_SS2D(d_model=self.path_proj_dim,type_layer=type_layer)
        self.AAM_text = MM_SS2D(d_model=self.path_proj_dim,type_layer=type_layer)

        self.lr_alpha = lr_alpha
        


    def forward_no_loss(self, x_path, x_text, x_diff_path, x_diff_text, return_attn=False):
        """
        Args:
            x_path: (B, numOfproto, in_dim) in_dim = [prob, mean, cov] (If OT, prob will be uniform, cov will be none)
            x_omics:
            return_attn:

        Returns:

        """
        device = x_path.device

        h_text = self.text_proj_net(x_text)
        h_path = self.path_proj_net(x_path)

        h_diff_path = self.diff_path_proj_net(x_diff_path)
        h_diff_text = self.diff_text_proj_net(x_diff_text)

        if self.histo_embedding is not None:    
            arr = []
            for idx in range(len(h_path)):
                arr.append(torch.cat([h_path[idx:idx + 1], self.histo_embedding.to(device)], dim=-1))
            h_path = torch.cat(arr, dim=0)
        
        #AAM
        h_path,h_diff_path = self.AAM_path(h_path,h_diff_path)
        h_text,h_diff_text = self.AAM_text(h_text,h_diff_text)

        tokens_p = torch.cat([h_path, h_diff_path], dim=1) 
        tokens_t = torch.cat([h_text, h_diff_text], dim=1) 

        h_path, h_omic = tokens_p, tokens_t
        #
        # Cross-modal optimal transport
        #
        p_to_h_coattn = self.mot(h_path, h_omic).float()
        h_to_p_coattn = self.mot(h_omic, h_path).float()

        h_path_coattn = torch.matmul(p_to_h_coattn, h_path)
        h_omic_coattn = torch.matmul(h_to_p_coattn, h_omic)

        tokens = torch.cat([ h_omic_coattn,h_path_coattn ], dim=1)  
        tokens = self.identity(tokens)

        # Pass the token set through co-attention network
        mm_embed = self.coattn(tokens)

        # ---> aggregate
        paths_postSA_embed = mm_embed[:, :self.num_text_proto, :]
        paths_postSA_embed = torch.mean(paths_postSA_embed, dim=1)

        wsi_postSA_embed = mm_embed[:, self.num_text_proto:, :]

        if self.histo_model == 'mil':
            wsi_postSA_embed = torch.mean(wsi_postSA_embed, dim=1)  # For non-prototypes, we just take the mean
        else:
            wsi_postSA_embed = agg_histo(wsi_postSA_embed, self.histo_agg)

        if self.modality == 'histo':    # Just use histo for prediction
            embedding = wsi_postSA_embed
        elif self.modality == 'gene':   # Just use gene for prediction
            embedding = paths_postSA_embed
        else:   # Use both modalities
            embedding = torch.cat([paths_postSA_embed, wsi_postSA_embed], dim=1)  # ---> both branches

        logits = self.classifier(embedding)
        out = {'logits': logits}
        

        #ibd
        out['h_path'],out['h_diff_path']= h_path,h_diff_path
        out['h_text'],out['h_diff_text']= h_text,h_diff_text

        out['histology'], out['pathways'] =torch.mean(tokens_p, dim=1),torch.mean(tokens_t, dim=1)
        out['global_embed'] = torch.mean(torch.cat([mm_embed[:, :self.num_text_proto, :],mm_embed[:, self.num_text_proto:, :]], dim=1), dim=1)

        return out

    def forward(self, h, model_kwargs={},text_emb=None):

        attn_mask = model_kwargs['attn_mask']
        label = model_kwargs['label']
        censorship = model_kwargs['censorship']
        loss_fn = model_kwargs['loss_fn']

        x_diff_t,x_diff_p,x_gs =h[...,:768],h[...,768:768*2],h[...,768*2:]
        out = self.forward_no_loss(x_path=x_gs,x_text=text_emb,x_diff_path=x_diff_p,x_diff_text=x_diff_t)
        logits = out['logits']

        results_dict, log_dict = process_surv(logits, label, censorship, loss_fn)
        
        histology, pathways, global_embed =out['histology'], out['pathways'], out['global_embed']

        mimin = self.CLUB(histology, pathways, global_embed)
        mimin_loss = self.CLUB.learning_loss(histology, pathways, global_embed)

        mmd_p= batch_mmd( out['h_path'],out['h_diff_path']).mean()
        mmd_t= batch_mmd( out['h_text'],out['h_diff_text']).mean()
        
        results_dict['loss'] = results_dict['loss'] + self.lr_alpha*(mimin_loss+ mimin) + self.lr_alpha*(mmd_p+mmd_t)

        return results_dict, log_dict



