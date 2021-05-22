import time
from options.train_options import TrainOptions
from data.my_data_loader import OccFaceImageMixLoaderV2
from torchvision import transforms

from models.base_model import ModelFactory
from utils.tb_visualizer import TBVisualizer
from collections import OrderedDict
import os
import torch.utils.data as torch_data
from utils.util import *
import torch

def loaderAndResize(path):

    return Image.open(path).resize((128, 128))

def loader(path):

    return Image.open(path)

class Train:
    def __init__(self):
        self._opt = TrainOptions().parse()

        self.train_transform = transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])
        ])

        self._dataset_train = torch_data.DataLoader(dataset=OccFaceImageMixLoaderV2(
                                            self._opt.data_root, 
                                            self._opt.train_list_wo_g_list,
                                            self._opt.occlusions_root,
                                            self._opt.occlusions_list,
                                            self._opt.data_root, 
                                            self._opt.train_list_w_g_list,
                                            transform=self.train_transform),
                                            batch_size=self._opt.batch_size, 
                                            num_workers = self._opt.n_threads_train,
                                            shuffle=True, drop_last=True)

        self._dataset_test = torch_data.DataLoader(dataset=OccFaceImageMixLoaderV2(
                                            self._opt.data_root, 
                                            self._opt.test_list_wo_g_list,
                                            self._opt.occlusions_root,
                                            self._opt.occlusions_list,
                                            self._opt.data_root, 
                                            self._opt.test_list_w_g_list,
                                            transform=self.test_transform),
                                            batch_size=self._opt.batch_size,
                                            drop_last=True)

        self._dataset_train_size = len(self._dataset_train)
        self._dataset_test_size = len(self._dataset_test)

        print('#train images = %d' % self._dataset_train_size)
        print('#test images = %d' % self._dataset_test_size)

        self._model = ModelFactory.get_by_name(self._opt.model, self._opt)
        self._tb_visualizer = TBVisualizer(self._opt)

        self._train()

    def _train(self):

        self._total_steps = self._opt.load_epoch * self._dataset_train_size
        self._iters_per_epoch = self._dataset_train_size 
        self._last_display_time = None
        self._last_save_latest_time = None
        self._last_print_time = time.time()

        for i_epoch in range(self._opt.load_epoch + 1, 
                                self._opt.nepochs_no_decay + self._opt.nepochs_decay + 1):
            epoch_start_time = time.time()

            self._train_epoch(i_epoch)

            print('saving the model at the end of epoch %d, iters %d' % (i_epoch, self._total_steps))
            self._model.save(i_epoch)

            time_epoch = time.time() - epoch_start_time
            print('End of epoch %d / %d \t Time Taken: %d sec (%d min or %d h)' %
                  (i_epoch, self._opt.nepochs_no_decay + self._opt.nepochs_decay, time_epoch,
                   time_epoch / 60, time_epoch / 3600))

            if i_epoch > self._opt.nepochs_no_decay:
                self._model.update_learning_rate()

    def _train_epoch(self, i_epoch):
        
        epoch_iter = 0
        self._model.set_train()
        for i_train_batch, train_batch in enumerate(self._dataset_train):

            iter_start_time = time.time()

            do_visuals = \
                self._last_display_time is None or \
                    time.time() - self._last_display_time > self._opt.display_freq_s
            do_print_terminal = \
                do_visuals or time.time() - self._last_print_time > self._opt.print_freq_s

            train_generator = \
                do_visuals or ((i_train_batch+1) % self._opt.train_G_every_n_iterations == 0)

            train_batch_split = {}

            if (i_train_batch + 1) % self._opt.fake_real_rate == 0:
                train_batch_split['none_occ_img'] = train_batch['face_w_syn_occ_img_GT']
                train_batch_split['none_occ_attr'] = train_batch['face_w_syn_occ_attr']


                train_batch_split['occ_img'] = train_batch['face_w_occ_img']
                train_batch_split['occ_attr'] = train_batch['face_w_occ_attr']


                train_batch_split['none_occ_img_adv'] = train_batch['face_wo_occ_img_adv']  
                train_batch_split['none_occ_attr_adv'] = train_batch['face_wo_occ_attr_adv']

                has_GT_flag = False
                has_attr_flag = True        
            else:
                train_batch_split['none_occ_img'] = train_batch['face_w_syn_occ_img_GT']
                train_batch_split['none_occ_attr'] = train_batch['face_w_syn_occ_attr']

                train_batch_split['occ_img'] = train_batch['face_w_syn_occ_img']
                train_batch_split['occ_attr'] = train_batch['face_w_syn_occ_attr']

                train_batch_split['none_occ_img_adv'] = train_batch['face_wo_occ_img_adv']  
                train_batch_split['none_occ_attr_adv'] = train_batch['face_wo_occ_attr_adv']

                has_GT_flag = True
                has_attr_flag = True

            # print(has_GT_flag, has_attr_flag)
            self._model.set_input(train_batch_split)
            self._model.optimize_parameters(keep_data_for_visuals=do_visuals, 
                                                train_generator=train_generator, 
                                                has_GT=has_GT_flag, has_attr=has_attr_flag)   
        
            if do_print_terminal:

                self._display_terminal(iter_start_time, i_epoch, 
                                            i_train_batch, do_visuals, 
                                            has_GT=has_GT_flag, has_attr=has_attr_flag)

                self._last_print_time = time.time()

            if do_visuals:
                self._display_visualizer_train(i_epoch,
                                                self._total_steps, 
                                                has_GT=has_GT_flag, 
                                                has_attr=has_attr_flag)
                self._display_visualizer_val(i_epoch, 
                                                self._total_steps, 
                                                has_GT=False, 
                                                has_attr=True)
                self._last_display_time = time.time()


            self._total_steps += self._opt.batch_size
            epoch_iter += self._opt.batch_size

            if self._last_save_latest_time is None or \
                time.time() - self._last_save_latest_time > self._opt.save_latest_freq_s:
                
                print('saving the latest model (epoch %d, total_steps %d)' % (i_epoch, self._total_steps))
                self._model.save(i_epoch)
                self._last_save_latest_time = time.time()


    def _display_terminal(self, iter_start_time, i_epoch, i_train_batch, visuals_flag, has_GT, has_attr):
        
        errors = self._model.get_current_errors(has_GT, has_attr)
        t = (time.time() - iter_start_time) / self._opt.batch_size
        
        self._tb_visualizer.print_current_train_errors(i_epoch, i_train_batch, 
                                                        self._iters_per_epoch, 
                                                        errors, t, visuals_flag)

    def _display_visualizer_train(self, i_epoch, total_steps, has_GT, has_attr):
        
        if has_GT == True and has_attr == True:
            flag = '_w_GT_w_attr'
        if has_GT == False and has_attr == True:
            flag = '_wo_GT_w_attr'

        

        self._tb_visualizer.display_current_results(self._model.get_current_visuals(), 
                                                        i_epoch, total_steps, is_train=True, 
                                                        save_visuals=True, flag=flag)
        self._tb_visualizer.plot_scalars(self._model.get_current_errors(has_GT, has_attr), 
                                            total_steps, is_train=True)
        self._tb_visualizer.plot_scalars(self._model.get_current_scalars(), 
                                            total_steps, is_train=True)

    def _display_visualizer_val(self, i_epoch, total_steps, has_GT, has_attr):
        
        if has_GT == True and has_attr == True:
            flag = '_w_GT_w_attr'
        if has_GT == False and has_attr == True:
            flag = '_wo_GT_w_attr'
        
        # print(has_GT, has_attr, flag)

        self._model.set_eval()

        for i_val_batch, val_batch in enumerate(self._dataset_test):

            if i_val_batch == self._opt.num_iters_validate:
                break
            
            val_batch_split = {}

            val_batch_split['none_occ_img'] = val_batch['face_w_syn_occ_img_GT']
            val_batch_split['none_occ_attr'] = val_batch['face_w_syn_occ_attr']

            val_batch_split['occ_img'] = val_batch['face_w_occ_img']
            val_batch_split['occ_attr'] = val_batch['face_w_occ_attr']

            val_batch_split['none_occ_img_adv'] = val_batch['face_wo_occ_img_adv']  
            val_batch_split['none_occ_attr_adv'] = val_batch['face_wo_occ_attr_adv']

            self._model.set_input(val_batch_split)
            self._model.forward(keep_data_for_visuals=True)

        self._tb_visualizer.display_current_results(self._model.get_current_visuals(), 
                                                    i_epoch, total_steps, is_train=False, 
                                                    save_visuals=True, flag=flag)

        self._model.set_train()


if __name__ == "__main__":
    Train()
