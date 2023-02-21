import sys
import os
import os.path as osp
import pdb

import json
import tqdm
import numpy as np
from dirtorch.utils.path_utils import get_data_root, get_embedding_root, get_embedding_subfolder, get_model_root, get_results_root
import torch
import torch.nn.functional as F

from dirtorch.utils.convenient import mkdir
from dirtorch.utils import common
from dirtorch.utils.common import tonumpy, matmul, pool
from dirtorch.utils.pytorch_loader import get_loader
import dirtorch.nets as nets
import dirtorch.datasets as datasets
import dirtorch.datasets.downloader as dl

import pickle as pkl
import hashlib
import pandas as pd
import os
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def expand_descriptors(descs, db=None, alpha=0, k=0):
    assert k >= 0 and alpha >= 0, 'k and alpha must be non-negative'
    if k == 0:
        return descs
    descs = tonumpy(descs)
    n = descs.shape[0]
    db_descs = tonumpy(db if db is not None else descs)

    sim = matmul(descs, db_descs)
    if db is None:
        sim[np.diag_indices(n)] = 0

    idx = np.argpartition(sim, int(-k), axis=1)[:, int(-k):]
    descs_aug = np.zeros_like(descs)
    for i in range(n):
        new_q = np.vstack([db_descs[j, :] * sim[i, j]**alpha for j in idx[i]])
        new_q = np.vstack([descs[i], new_q])
        new_q = np.mean(new_q, axis=0)
        descs_aug[i] = new_q / np.linalg.norm(new_q)

    return descs_aug


def extract_image_features(dataset, transforms, net, ret_imgs=False, same_size=False, flip=None,
                           desc="Extract feats...", iscuda=True, threads=8, batch_size=8):
    """ Extract image features for a given dataset.
        Output is 2-dimensional (B, D)
    """
    if not same_size:
        batch_size = 1
        old_benchmark = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = False

    print(dataset)

    loader = get_loader(dataset, trf_chain=transforms, preprocess=net.preprocess, iscuda=iscuda,
                        output=['img'], batch_size=batch_size, threads=threads, shuffle=False)

    if hasattr(net, 'eval'):
        net.eval()

    tocpu = (lambda x: x.cpu()) if ret_imgs == 'cpu' else (lambda x: x)
    
    img_names = []
    img_feats = []
    trf_images = []
    with torch.no_grad():
        for temp_inputs in tqdm.tqdm(loader, desc, total=1+(len(dataset)-1)//batch_size):
            inputs = temp_inputs[0]
            image_name = temp_inputs[1]
            img_names.append(image_name[0])
            imgs = inputs[0]
            for i in range(len(imgs)):
                if flip and flip.pop(0):
                    imgs[i] = imgs[i].flip(2)
            imgs = common.variables(inputs[:1], net.iscuda)[0]
            desc = net(imgs)
            if ret_imgs:
                trf_images.append(tocpu(imgs.detach()))
            del imgs
            del inputs
            if len(desc.shape) == 1:
                desc.unsqueeze_(0)
            img_feats.append(desc.detach())

    img_feats = torch.cat(img_feats, dim=0)
    if len(img_feats.shape) == 1:
        img_feats.unsqueeze_(0)

    if not same_size:
        torch.backends.cudnn.benchmark = old_benchmark
        
    if ret_imgs:
        if same_size:
            trf_images = torch.cat(trf_images, dim=0)
        return trf_images, img_feats
    return img_feats, img_names

# This time it's num_image x embeddings_size
def generate_embedding_file(image_list, embeddings,file_path):
    list_embedding = embeddings.tolist()
    df = pd.DataFrame( {'image_name': image_list, 'embedding' : list_embedding})
    df.to_csv(file_path, index=False)

global bdescs
global qdescs

def eval_model(db, net, trfs, pooling='mean', gemp=3, detailed=False, whiten=None,
               aqe=None, adba=None, threads=8, batch_size=16, save_feats=None,
               load_feats=None, dbg=()):
    """ Evaluate a trained model (network) on a given dataset.
    The dataset is supposed to contain the evaluation code.
    """
    print("\n>> Evaluation...")
    query_db = db.get_query_db()

    # extract DB feats
    bdescs = []
    qdescs = []

    if not load_feats:
        trfs_list = [trfs] if isinstance(trfs, str) else trfs

        for trfs in trfs_list:
            kw = dict(iscuda=net.iscuda, threads=threads, batch_size=batch_size, same_size='Pad' in trfs or 'Crop' in trfs)
            (db_features,db_image_names) = extract_image_features(db, trfs, net, desc="DB", **kw)
            bdescs.append(db_features)

            # extract query feats
            (q_features, q_image_names) = extract_image_features(query_db, trfs, net, desc="query", **kw)
            qdescs.append(bdescs[-1] if db is query_db else q_features)

        # pool from multiple transforms (scales)
        bdescs = F.normalize(pool(bdescs, pooling, gemp), p=2, dim=1)
        qdescs = F.normalize(pool(qdescs, pooling, gemp), p=2, dim=1)
    else:
        bdescs = np.load(os.path.join(load_feats, 'feats.bdescs.npy'))
        if query_db is not db:
            qdescs = np.load(os.path.join(load_feats, 'feats.qdescs.npy'))
        else:
            qdescs = bdescs

    if save_feats:
        mkdir(save_feats)
        np.save(os.path.join(save_feats, 'feats.bdescs.npy'), bdescs.cpu().numpy())
        if query_db is not db:
            np.save(os.path.join(save_feats, 'feats.qdescs.npy'), qdescs.cpu().numpy())

    if whiten is not None:
        bdescs = common.whiten_features(tonumpy(bdescs), net.pca, **whiten)
        qdescs = common.whiten_features(tonumpy(qdescs), net.pca, **whiten)

    if adba is not None:
        bdescs = expand_descriptors(bdescs, **args.adba)
    if aqe is not None:
        qdescs = expand_descriptors(qdescs, db=bdescs, **args.aqe)

    dataset_name = args.dataset.lower()
    dataset_embedding_path = os.path.join(get_embedding_subfolder(), "{}-dataset-features.csv".format(dataset_name))
    query_embedding_path = os.path.join(get_embedding_subfolder(), "{}-query-features.csv".format(dataset_name))

    generate_embedding_file(db_image_names, bdescs, dataset_embedding_path)
    generate_embedding_file(q_image_names, qdescs, query_embedding_path)

    scores = matmul(qdescs, bdescs)

    res = {}
    
    df = pd.DataFrame(columns = ['query_path','results_path','query_emb','result_emb','scores'])

    try:
        for q,s in enumerate(scores):
            row = db.eval_query_AP(q, s, bdescs,qdescs[q], q_image_names[q],db_image_names)[1]
            # row = db.eval_query_top_k(q, s, bdescs,qdescs[q], q_image_names[q],db_image_names)[1]
            df = pd.concat([df, pd.DataFrame.from_records([row])])
        
        top_100_results_path = os.path.join(get_results_root(), "{}-top-100-results-and-scores.csv".format(dataset_name))
        df.to_csv(top_100_results_path,index=False)
        return
    except:
        raise NotImplemented("Not implemented")

def load_model(path, iscuda):
    checkpoint = common.load_checkpoint(path, iscuda)
    net = nets.create_model(pretrained="", **checkpoint['model_options'])
    net = common.switch_model_to_cuda(net, iscuda, checkpoint)
    net.load_state_dict(checkpoint['state_dict'])
    net.preprocess = checkpoint.get('preprocess', net.preprocess)
    if 'pca' in checkpoint:
        net.pca = checkpoint.get('pca')
    return net


if __name__ == '__main__':

    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    if not os.path.exists(get_data_root()):
        os.makedirs(get_data_root())

    if not os.path.exists(get_embedding_root()):
        os.makedirs(get_embedding_root())
        os.makedir(get_embedding_root())
    
    if not os.path.exists(get_model_root()):
        os.makedirs(get_model_root())
    
    if not os.path.exists(get_results_root()):
        os.makedirs(get_results_root())
        os.makedirs(get_embedding_subfolder())



    import argparse
    parser = argparse.ArgumentParser(description='Evaluate a model')

    parser.add_argument('--dataset', '-d', type=str, required=True, help='Command to load dataset')
    #parser.add_argument('--checkpoint', type=str, required=False, help='path to weights')

    parser.add_argument('--trfs', type=str, required=False, default='', nargs='+', help='test transforms (can be several)')
    parser.add_argument('--pooling', type=str, default="gem", help='pooling scheme if several trf chains')
    parser.add_argument('--gemp', type=int, default=3, help='GeM pooling power')

    parser.add_argument('--out-json', type=str, default="", help='path to output json')
    parser.add_argument('--detailed', action='store_true', help='return detailed evaluation')
    parser.add_argument('--save-feats', type=str, default="", help='path to output features')
    parser.add_argument('--load-feats', type=str, default="", help='path to load features from')

    parser.add_argument('--threads', type=int, default=8, help='number of thread workers')
    parser.add_argument('--gpu', type=int, default=0, nargs='+', help='GPU ids')
    parser.add_argument('--dbg', default=(), nargs='*', help='debugging options')
    # post-processing
    parser.add_argument('--whiten', type=str, default='Landmarks_clean', help='applies whitening')

    parser.add_argument('--aqe', type=int, nargs='+', help='alpha-query expansion paramenters')
    parser.add_argument('--adba', type=int, nargs='+', help='alpha-database augmentation paramenters')

    parser.add_argument('--whitenp', type=float, default=0.25, help='whitening power, default is 0.5 (i.e., the sqrt)')
    parser.add_argument('--whitenv', type=int, default=None, help='number of components, default is None (i.e. all components)')
    parser.add_argument('--whitenm', type=float, default=1.0, help='whitening multiplier, default is 1.0 (i.e. no multiplication)')

    os.environ['DB_ROOT']  = get_data_root()

    checkpoint_path = os.path.join(os.path.join(get_model_root(),'networks'),'Resnet-101-AP-GeM.pt')

    if(not(os.path.exists(checkpoint_path))):
        raise Exception('Checkpoint {} does not exist'.format(checkpoint_path))

    
    args = parser.parse_args()
    args.iscuda = common.torch_set_gpu(args.gpu)
    if args.aqe is not None:
        args.aqe = {'k': args.aqe[0], 'alpha': args.aqe[1]}
    if args.adba is not None:
        args.adba = {'k': args.adba[0], 'alpha': args.adba[1]}

    dl.download_dataset(args.dataset)

    dataset = datasets.create(args.dataset)
    print("Test dataset:", dataset)

    net = load_model(checkpoint_path, args.iscuda)

    if args.whiten:
        net.pca = net.pca[args.whiten]
        args.whiten = {'whitenp': args.whitenp, 'whitenv': args.whitenv, 'whitenm': args.whitenm}
    else:
        net.pca = None
        args.whiten = None

    # Evaluate
    res = eval_model(dataset, net, args.trfs, pooling=args.pooling, gemp=args.gemp, detailed=args.detailed,
                     threads=args.threads, dbg=args.dbg, whiten=args.whiten, aqe=args.aqe, adba=args.adba,
                     save_feats=args.save_feats, load_feats=args.load_feats)