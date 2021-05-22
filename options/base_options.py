import argparse
import os
from utils import util
import torch

class BaseOptions():
    def __init__(self):
        self._parser = argparse.ArgumentParser()
        self._initialized = False

    def initialize(self):

        self._parser.add_argument('--fake_real_rate', type=int, default=10, help='')

        self._parser.add_argument('--mask_root', type=str, 
                                    default='/home/jccai/ssd/celebA_oagan/face_mask_128', 
                                    help='')
        self._parser.add_argument('--mask_train_list', type=str, 
                                    default='/home/jccai/ssd/celebA_oagan/face_mask_train_list.pickle', 
                                    help='')
        self._parser.add_argument('--mask_test_list', type=str, 
                                    default='/home/jccai/ssd/celebA_oagan/face_mask_test_list.pickle', 
                                    help='')

        self._parser.add_argument('--occlusions_root', type=str, 
                                    default='/home/jccai/ssd/celebA_oagan/occ', 
                                    help='')
        self._parser.add_argument('--occlusions_list', type=str, 
                                    default='/home/jccai/ssd/celebA_oagan/occ_list.pickle', 
                                    help='')

        self._parser.add_argument('--data_root', type=str, 
                                    default='/home/jccai/ssd/celebA_oagan/celebA_crop_144', 
                                    help='')
        self._parser.add_argument('--train_list_w_g_list', type=str, 
                                    default='/home/jccai/ssd/celebA_oagan/celebA_w_glass_train_list_repeat.pickle', 
                                    help='path to train list with glass')
        self._parser.add_argument('--test_list_w_g_list', type=str, 
                                    default='/home/jccai/ssd/celebA_oagan/celebA_w_glass_test_list_repeat.pickle', 
                                    help='')

        self._parser.add_argument('--train_list_wo_g_list', type=str, 
                                    default='/home/jccai/ssd/celebA_oagan/celebA_wo_glass_train_list_repeat.pickle', 
                                    help='')
        self._parser.add_argument('--test_list_wo_g_list', type=str, 
                                    default='/home/jccai/ssd/celebA_oagan/celebA_wo_glass_test_list_repeat.pickle', 
                                    help='')

        self._parser.add_argument('--load_epoch', type=int, default=-1, help='which epoch to load? set to -1 to use latest cached model')
        self._parser.add_argument('--batch_size', type=int, default=48, help='input batch size')
        self._parser.add_argument('--image_size', type=int, default=128, help='input image size')
        self._parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self._parser.add_argument('--name', type=str, default='experiment_38_alternate_training_-_10pow(2)_landmark_sig_10_wo_hash', help='name of the experiment. It decides where to store samples and models')
        self._parser.add_argument('--model', type=str, default='FOAFCGAN_alternate_training', help='model to run')
        self._parser.add_argument('--n_threads_test', default=1, type=int, help='# threads for loading data')
        self._parser.add_argument('--checkpoints_dir', type=str, default='../OA-saved/checkpoints', help='models are saved here')
        self._initialized = True

    def parse(self):

        if not self._initialized:
            self.initialize()

        self._opt = self._parser.parse_args()

        self._opt.is_train = self.is_train

        self._set_and_check_load_epoch()

        self._get_set_gpus()

        args = vars(self._opt)

        self._print(args)

        self._save(args)

        return self._opt

    def _set_and_check_load_epoch(self):

        models_dir = os.path.join(self._opt.checkpoints_dir, self._opt.name)
        if os.path.exists(models_dir):
            if self._opt.load_epoch == -1:
                load_epoch = 0
                for file in os.listdir(models_dir):
                    if file.startswith("net_epoch_"):
                        load_epoch = max(load_epoch, int(file.split('_')[2]))
                self._opt.load_epoch = load_epoch
            else:
                found = False
                for file in os.listdir(models_dir):
                    if file.startswith("net_epoch_"):
                        found = int(file.split('_')[2]) == self._opt.load_epoch
                        if found: break
                assert found, 'Model for epoch %i not found' % self._opt.load_epoch
        else:
            assert self._opt.load_epoch < 1, 'Model for epoch %i not found' % self._opt.load_epoch
            self._opt.load_epoch = 0

    def _get_set_gpus(self):

        str_ids = self._opt.gpu_ids.split(',')
        self._opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self._opt.gpu_ids.append(id)

    def _print(self, args):

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

    def _save(self, args):
        
        expr_dir = os.path.join(self._opt.checkpoints_dir, self._opt.name)
        print(expr_dir)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt_%s.txt' % ('train' if self.is_train else 'test'))
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
