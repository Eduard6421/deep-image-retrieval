import os
import os.path as osp

from dirtorch.utils.path_utils import get_data_root


def download_dataset(dataset):
    if not os.path.isdir(get_data_root()):
        os.makedirs(get_data_root())

    dataset = dataset.lower()
    print(dataset)
    if dataset in ('oxford5k', 'roxford5k','roxford5k_drift'):
        src_dir = 'http://www.robots.ox.ac.uk/~vgg/data/oxbuildings'
        dl_files = ['oxbuild_images.tgz']
        dir_name = 'oxford5k'
    elif dataset in ('paris6k', 'rparis6k','rparis6k_drift'):
        src_dir = 'http://www.robots.ox.ac.uk/~vgg/data/parisbuildings'
        dl_files = ['paris_1.tgz', 'paris_2.tgz']
        dir_name = 'paris6k'
    elif dataset in ['pascalvoc','pascalvoc_70','pascalvoc_140','pascalvoc_350','pascalvoc_700','pascalvoc_1400','pascalvoc_700_no_bbx',
                    'pascalvoc_700_medium','pascalvoc_700_train','pascalvoc_700_medium_train','pascalvoc_700_no_bbx_train',
                    'pascalvoc_700_drift','pascalvoc_700_medium_drift','pascalvoc_700_no_bbx_drift'
                    ]:
        dir_name = 'pascalvoc'
        dl_files =['pascal.tgz']
        src_dir = 'no data here'
    elif dataset in ['caltech101_70','caltech101_350','caltech101_700','caltech101_1400','caltech101_700_train','caltech101_700_drift']:
        dir_name = 'caltech101'
        dl_files = ['caltech.tgz']
        src_dir = 'no data here'
    else:
        print(dataset)
        raise ValueError('Unknown dataset: {}!'.format(dataset))

    dst_dir = os.path.join(get_data_root(), dir_name, 'jpg')
    if not os.path.isdir(dst_dir):
        print('>> Dataset {} directory does not exist. Creating: {}'.format(dataset, dst_dir))
        os.makedirs(dst_dir)
        for dli in range(len(dl_files)):
            dl_file = dl_files[dli]
            src_file = os.path.join(src_dir, dl_file)
            dst_file = os.path.join(dst_dir, dl_file)
            print('>> Downloading dataset {} archive {}...'.format(dataset, dl_file))
            os.system('wget {} -O {}'.format(src_file, dst_file))
            print('>> Extracting dataset {} archive {}...'.format(dataset, dl_file))
            # create tmp folder
            dst_dir_tmp = os.path.join(dst_dir, 'tmp')
            os.system('mkdir {}'.format(dst_dir_tmp))
            # extract in tmp folder
            os.system('tar -zxf {} -C {}'.format(dst_file, dst_dir_tmp))
            # remove all (possible) subfolders by moving only files in dst_dir
            os.system('find {} -type f -exec mv -i {{}} {} \\;'.format(dst_dir_tmp, dst_dir))
            # remove tmp folder
            os.system('rm -rf {}'.format(dst_dir_tmp))
            print('>> Extracted, deleting dataset {} archive {}...'.format(dataset, dl_file))
            os.system('rm {}'.format(dst_file))

    gnd_src_dir = os.path.join('http://cmp.felk.cvut.cz/cnnimageretrieval/data', 'test', dataset)
    gnd_dst_dir = os.path.join(get_data_root(), dir_name)
    gnd_dl_file = 'gnd_{}.pkl'.format(dataset)
    gnd_src_file = os.path.join(gnd_src_dir, gnd_dl_file)
    gnd_dst_file = os.path.join(gnd_dst_dir, gnd_dl_file)
    if not os.path.exists(gnd_dst_file):
        print('>> Downloading dataset {} ground truth file...'.format(dataset))
        os.system('wget {} -O {}'.format(gnd_src_file, gnd_dst_file))
