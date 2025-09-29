import os
import re
import torch
from utils.file_utils import save_pkl, load_pkl
from os.path import join as j_
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def create_downstream_model(args, mode='classification', config_dir='./configs'):
    """
    Create downstream modles for classification or survival
    """
    config_path = os.path.join(config_dir, args.model_config, 'config.json')
    # assert os.path.exists(config_path), f"Config path {config_path} doesn't exist!"
    
    model_config = args.model_config
    model_type = args.model_type

    if 'IndivMLPEmb' in model_config:
        update_dict = {'in_dim': args.in_dim,
                       'p': args.out_size,
                       'out_type': args.out_type,
                       }

    elif model_type == 'DeepAttnMIL':
        update_dict = {'in_dim': args.in_dim,
                       'out_size': args.out_size,
                       'load_proto': args.load_proto,
                       'fix_proto': args.fix_proto,
                       'proto_path': args.proto_path}
    else:
        # update_dict = {'in_dim': args.in_dim}
        update_dict ={}

    if mode == 'classification':
        update_dict.update({'n_classes': args.n_classes})
    elif mode == 'survival':
        if args.loss_fn == 'nll':
            update_dict.update({'n_classes': args.n_label_bins})
        elif args.loss_fn == 'cox':
            update_dict.update({'n_classes': 1})
        elif args.loss_fn == 'rank':
            update_dict.update({'n_classes': 1})
    else:
        raise NotImplementedError(f"Not implemented for {mode}...")
    
    if model_type == 'ABMIL':
        update_dict.update({'in_dim': args.in_dim})
        config = ABMILConfig.from_pretrained(config_path, update_dict=update_dict)
        model = ABMIL(config=config, mode=mode)


    elif model_type.startswith('coattn_IBDMMD_transformer_256_net_indiv'):
        from src.mil_models.model_multimodal_IBDMMD import coattn
        # alpha
        match = re.search(r'_alpha_(\d*\.\d+)', model_type)
        lr_alpha = float(match.group(1)) if match else 0.05

        # beta
        match = re.search(r'_beta_(\d*\.\d+)', model_type)
        lr_beta = float(match.group(1)) if match else 0.05
        # Search for dim
        match_dim = re.search(r'_dim_(\d+)$', model_type)  
        dim = int(match_dim.group(1)) if match_dim else 256  
        # Search for proto_dim
        match_dim = re.search(r'_proto_(\d+)$', model_type)  
        proto_dim = int(match_dim.group(1)) if match_dim else 16  
        print('************model specific hyperparameter************')
        print('lr_alpha:', lr_alpha, 'path_proj_dim:', dim, 'proto_dim:', proto_dim, 'lr_beta:', lr_beta)
        model = coattn(dropout=args.in_dropout, num_classes=update_dict['n_classes'],histo_model='panther',path_proj_dim=dim,type_layer='Transformer',net_indiv=True,
                       lr_alpha=lr_alpha,numOfproto=proto_dim,lr_beta=lr_beta) 

    else:
        raise NotImplementedError

    return model


def prepare_emb(datasets, args, mode='classification'):
    """
    Slide representation construction with patch feature aggregation trained in unsupervised manner
    """
   
    ### Preparing file path for saving embeddings
    print('\nConstructing unsupervised slide embedding...', end=' ')
    embeddings_kwargs = {
        'feats': args.data_source[0].split('/')[-3],
        'model_type': args.model_type,
        'out_size': args.n_proto
    }

    # Create embedding path
    fpath = "{feats}_{model_type}_embeddings_proto_{out_size}".format(**embeddings_kwargs)
    if args.model_type == 'PANTHER':
        DIEM_kwargs = {'tau': args.tau, 'out_type': args.out_type, 'eps': args.ot_eps, 'em_step': args.em_iter}
        name = '_{out_type}_em_{em_step}_eps_{eps}_tau_{tau}'.format(**DIEM_kwargs)
        fpath += name
    elif args.model_type == 'OT':
        OTK_kwargs = {'out_type': args.out_type, 'eps': args.ot_eps}
        name = '_{out_type}_eps_{eps}'.format(**OTK_kwargs)
        fpath += name
    embeddings_fpath = j_(args.split_dir, 'embeddings', fpath+'.pkl')
    
    ### Load existing embeddings if already created
    if os.path.isfile(embeddings_fpath):
        embeddings = load_pkl(embeddings_fpath)
        for k, loader in datasets.items():
            print(f'\n\tEmbedding already exists! Loading {k}', end=' ')
            loader.dataset.X, loader.dataset.y = embeddings[k]['X'], embeddings[k]['y']
            ############# only use for nsclc_NLST
            if 'nsclc_NLST' in args.tags[0] and  k in [ 'val', 'test']:
                print('use nsclc_NLST embeddings')
                embeddings_tmp = load_pkl(j_(args.split_dir.replace('LUAD','NLST'), 'embeddings', 'nsclc_nlst_featCLIP_10x_256_PANTHER_embeddings_proto_16_allcat_em_1_eps_1.0_tau_1.0.pkl'))
                loader.dataset.X, loader.dataset.y = embeddings_tmp[k]['X'], embeddings_tmp[k]['y']
            ######################## end


    return datasets, embeddings_fpath