import os
import numpy as np
from collections import OrderedDict
import math
import matplotlib

import matplotlib.pyplot as plt
import PIL.Image as Image

import torch

from models.base_model import BaseModel
from models.modules.base_module import ModuleFactory
import utils.util as util
from models.modules.vgg import VGG16FeatureExtractor


class FOAFCGAN_alternate_training(BaseModel):

    def __init__(self, opt):

        super(FOAFCGAN_alternate_training, self).__init__(opt)

        self._name = 'FOAFCGAN_alternate_training_1'

        self._init_create_networks()

        if self._is_train:
            self._init_train_vars()

        if not self._is_train or self._opt.load_epoch > 0:
            self.load()

        self._init_prefetch_inputs()

        self._init_losses()
        

    def _init_create_networks(self):

        self._G = self._create_generator()
        self._G.init_weights()
        self._G = torch.nn.DataParallel(self._G, device_ids=[0])

        self._D = self._create_discriminator()
        self._D.init_weights()
        self._D = torch.nn.DataParallel(self._D, device_ids=[0])

        self._vgg = VGG16FeatureExtractor()
        self._vgg = torch.nn.DataParallel(self._vgg, device_ids=[0])

    def _create_generator(self):
        return ModuleFactory.get_by_name('generator_wgan')

    def _create_discriminator(self):
        return ModuleFactory.get_by_name('discriminator_wgan_cls')

    def _init_train_vars(self):

        self._current_lr_G = self._opt.lr_G
        self._current_lr_D = self._opt.lr_D

        self._optimizer_G = torch.optim.Adam(self._G.parameters(), lr=self._current_lr_G,
                                             betas=[self._opt.G_adam_b1, self._opt.G_adam_b2])
        self._optimizer_D = torch.optim.Adam(self._D.parameters(), lr=self._current_lr_D,
                                             betas=[self._opt.D_adam_b1, self._opt.D_adam_b2])

    def _init_prefetch_inputs(self):

        self._input_img_occ = \
            self._Tensor(self._opt.batch_size, 3, self._opt.image_size, self._opt.image_size)
        self._input_occ_attr = \
            self._Tensor(self._opt.batch_size, self._opt.attr_nc)

        self._input_img_none_occ = \
            self._Tensor(self._opt.batch_size, 3, self._opt.image_size, self._opt.image_size)
        self._input_none_occ_attr = \
            self._Tensor(self._opt.batch_size, self._opt.attr_nc)

        self._input_img_none_occ_adv = \
            self._Tensor(self._opt.batch_size, 3, self._opt.image_size, self._opt.image_size)
        self._input_none_occ_attr_adv = \
            self._Tensor(self._opt.batch_size, self._opt.attr_nc)

    def _init_losses(self):

        self._compute_loss_l1 = torch.nn.L1Loss()
        self._compute_loss_attr = torch.nn.MSELoss()

        # real and fake occluded face image loss
        self._loss_g_mask = self._Tensor([0])
        self._loss_g_masked_fake = self._Tensor([0])
        self._loss_g_mask_smooth = self._Tensor([0])
        # self._loss_g_mask_hash = self._Tensor([0])
        self._loss_g_attr = self._Tensor([0])
        self._loss_g_synth_smooth = self._Tensor([0])

        # fake occluded face image loss
        self._loss_g_vaild = self._Tensor([0])
        self._loss_g_hole = self._Tensor([0])
        self._loss_g_perceptual = self._Tensor([0])
        self._loss_g_style = self._Tensor([0])

        # d loss
        self._loss_d_attr = self._Tensor([0])
        self._loss_d_real = self._Tensor([0])
        self._loss_d_fake = self._Tensor([0])
        self._loss_d_gp = self._Tensor([0])

    def set_input(self, input):

        self._input_img_occ.resize_(input['occ_img'].size()).copy_(input['occ_img'])
        self._input_img_none_occ.resize_(input['none_occ_img'].size()).copy_(input['none_occ_img'])
        self._input_img_none_occ_adv.resize_(
            input['none_occ_img_adv'].size()).copy_(input['none_occ_img_adv'])

        if input['occ_attr'] is not None:
            self._input_occ_attr.resize_(
                input['occ_attr'].size()).copy_(input['occ_attr'])
        
        if input['none_occ_attr'] is not None:
            self._input_none_occ_attr.resize_(
                input['none_occ_attr'].size()).copy_(input['none_occ_attr'])

        if input['none_occ_attr_adv'] is not None:
            self._input_none_occ_attr_adv.resize_(
                input['none_occ_attr_adv'].size()).copy_(input['none_occ_attr_adv'])

        self._input_img_occ = self._input_img_occ.to(self._device)
        self._input_img_none_occ = self._input_img_none_occ.to(self._device)
        self._input_img_none_occ_adv = self._input_img_none_occ_adv.to(self._device)
        

        self._input_occ_attr = self._input_occ_attr.to(self._device)
        self._input_none_occ_attr = self._input_none_occ_attr.to(self._device)
        self._input_none_occ_attr_adv = self._input_none_occ_attr_adv.to(self._device)

    def set_train(self):

        self._G.train()
        self._D.train()
        self._is_train = True

    def set_eval(self):

        self._G.eval()
        self._is_train = False


    def forward(self, keep_data_for_visuals=False):

        if not self._is_train:

            im_occ = self._input_img_occ

            fake_img, fake_img_mask = self._G.forward(im_occ)
            fake_img_synthesis = fake_img_mask * im_occ + (1 - fake_img_mask) * fake_img

            if keep_data_for_visuals:

                self._vis_batch_occ_img = util.tensor2im(im_occ, idx=-1)
                self._vis_batch_fake_img = util.tensor2im(fake_img.data, idx=-1)
                self._vis_batch_fake_img_mask = util.tensor2maskim(fake_img_mask.data, idx=-1)
                self._vis_batch_fake_synthesis = util.tensor2im(fake_img_synthesis.data, idx=-1)
                self._vis_batch_none_occ_img = util.tensor2im(self._input_img_none_occ, idx=-1)


    def optimize_parameters(self, train_generator=True, keep_data_for_visuals=False, has_GT=False, has_attr=False):
        
        if self._is_train:
            self._B = self._input_img_occ.size(0)
            self._img_occ = self._input_img_occ
            self._img_none_occ = self._input_img_none_occ
            self._img_none_occ_adv = self._input_img_none_occ_adv

            self._none_occ_attr = self._input_none_occ_attr
            self._occ_attr = self._input_occ_attr
            self._occ_attr_adv = self._input_none_occ_attr_adv

            loss_D, fake_img_synthesis = self._forward_D(has_attr)
            self._optimizer_D.zero_grad()
            loss_D.backward()
            self._optimizer_D.step()

            loss_D_gp = self._gradinet_penalty_D(fake_img_synthesis)
            self._optimizer_D.zero_grad()
            loss_D_gp.backward()
            self._optimizer_D.step()

            if train_generator:
                loss_G = self._forward_G(keep_data_for_visuals, has_GT, has_attr)
                self._optimizer_G.zero_grad()
                loss_G.backward()
                self._optimizer_G.step()

    def _forward_G(self, keep_data_for_visuals, has_GT, has_attr):

        fake_img, fake_img_mask = self._G.forward(self._img_occ)
        fake_img_synthesis = fake_img_mask * self._img_occ + (1 - fake_img_mask) * fake_img

        if has_GT == True:

            fake_img_synthesis_feature = self._vgg(fake_img_synthesis)
            fake_img_feature = self._vgg(fake_img)
            gt_img_feature = self._vgg(self._img_none_occ)

            style = 0
            perceptual = 0

            for i in range(3):

                style += self._compute_loss_l1(self._compute_loss_gram_matrix(fake_img_feature[i]), 
                        self._compute_loss_gram_matrix(gt_img_feature[i]))
                style += self._compute_loss_l1(self._compute_loss_gram_matrix(fake_img_synthesis_feature[i]), 
                    self._compute_loss_gram_matrix(gt_img_feature[i]))

                perceptual += self._compute_loss_l1(fake_img_feature[i], gt_img_feature[i]) 
                perceptual += self._compute_loss_l1(fake_img_synthesis_feature[i], gt_img_feature[i])

            self._loss_g_style = style * self._opt.lambda_g_style
            self._loss_g_perceptual = perceptual * self._opt.lambda_g_perceptual

            target = (1 - fake_img_mask) * self._img_none_occ
            target = target.detach()
            self._loss_g_hole = self._compute_loss_l1((1 - fake_img_mask) * fake_img, target) * self._opt.lambda_g_hole
            
            target = fake_img_mask * self._img_none_occ
            target = target.detach()
            self._loss_g_vaild = self._compute_loss_l1(fake_img_mask * fake_img, target) * self._opt.lambda_g_valid
        
        # self._loss_g_mask_hash = -0.5 * torch.abs(fake_img_mask - 0.5).mean() * self._opt.lambda_g_hash

        d_fake_img_synthesis_prob, d_fake_img_attr = self._D.forward(fake_img_synthesis)
        
        if has_attr == True:
            self._loss_g_attr = self._compute_loss_attr(d_fake_img_attr, self._occ_attr) / self._B * self._opt.lambda_D_attr

        self._loss_g_synthesis_fake = self._compute_loss_D(d_fake_img_synthesis_prob, True) * self._opt.lambda_D_prob
        self._loss_g_mask = -torch.mean(fake_img_mask).pow(2) * self._opt.lambda_mask
        self._loss_g_mask_smooth = self._compute_loss_smooth(fake_img_mask) * self._opt.lambda_mask_smooth
        self._loss_g_synth_smooth = self._compute_loss_smooth(fake_img_synthesis) * self._opt.lambda_g_syhth_smooth
        
        if keep_data_for_visuals:

            self._vis_batch_occ_img = util.tensor2im(self._input_img_occ, idx=-1)
            self._vis_batch_fake_img = util.tensor2im(fake_img.data, idx=-1)
            self._vis_batch_fake_img_mask = util.tensor2maskim(fake_img_mask.data, idx=-1)
            self._vis_batch_fake_synthesis = util.tensor2im(fake_img_synthesis.data, idx=-1)
            self._vis_batch_none_occ_img = util.tensor2im(self._input_img_none_occ, idx=-1)

        if has_GT == True and has_attr == True:
            return self._loss_g_synthesis_fake + self._loss_g_mask + \
                self._loss_g_mask_smooth + self._loss_g_synth_smooth +\
                self._loss_g_vaild + self._loss_g_hole + \
                self._loss_g_perceptual + self._loss_g_style + \
                self._loss_g_attr  # + self._loss_g_mask_hash + \

        elif has_GT == False and has_attr == True:
            return self._loss_g_synthesis_fake + self._loss_g_mask + \
                    self._loss_g_mask_smooth + self._loss_g_synth_smooth +\
                    self._loss_g_attr # + self._loss_g_mask_hash

        elif has_GT == False and has_attr == False:
            return self._loss_g_synthesis_fake + self._loss_g_mask + \
                    self._loss_g_mask_smooth + self._loss_g_synth_smooth 
                    #+ self._loss_g_mask_hash
        else:
            raise NotImplementedError('Not existing has_GT = False and has_attr = True')
            return None

    def _forward_D(self, has_attr):

        d_real_img_prob, d_real_img_attr = self._D.forward(self._img_none_occ_adv)

        self._loss_d_real = self._compute_loss_D(d_real_img_prob, True) * self._opt.lambda_D_prob
        
        if has_attr:
            self._loss_d_attr = \
                self._compute_loss_attr(d_real_img_attr, 
                                            self._occ_attr_adv) / self._B * self._opt.lambda_D_attr


        fake_img, fake_img_mask = self._G.forward(self._img_occ)
        fake_img_synthesis = fake_img_mask * self._img_occ + (1 - fake_img_mask) * fake_img

        d_fake_img_prob, _ = self._D.forward(fake_img_synthesis.detach())
        self._loss_d_fake = self._compute_loss_D(d_fake_img_prob, False) * self._opt.lambda_D_prob

        if has_attr:
            return self._loss_d_real + self._loss_d_fake + self._loss_d_attr, fake_img_synthesis
        else:
            return self._loss_d_real + self._loss_d_fake, fake_img_synthesis


    def _gradinet_penalty_D(self, fake_img_synthesis):

        alpha = torch.rand(self._B, 1, 1, 1).expand_as(self._img_none_occ_adv).to(self._device)
        interpolated = alpha * self._img_none_occ_adv.data + (1 - alpha) * fake_img_synthesis.data
        interpolated.requires_grad = True
        interpolated_prob, _ = self._D(interpolated)

        grad = torch.autograd.grad(outputs=interpolated_prob,
                                   inputs=interpolated,
                                   grad_outputs=torch.ones(interpolated_prob.size()).to(self._device),
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        grad = grad.view(grad.size(0), -1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        self._loss_d_gp = torch.mean((grad_l2norm - 1) ** 2) * self._opt.lambda_D_gp

        return self._loss_d_gp

    def _compute_loss_D(self, estim, is_real):
        
        return -torch.mean(estim) if is_real else torch.mean(estim)

    def _compute_loss_smooth(self, mat):

        return torch.sum(torch.abs(mat[:, :, :, :-1] - mat[:, :, :, 1:])) + \
               torch.sum(torch.abs(mat[:, :, :-1, :] - mat[:, :, 1:, :]))

    def _compute_loss_gram_matrix(self, feat):

        (b, ch, h, w) = feat.size()
        feat = feat.view(b, ch, h * w)
        feat_t = feat.transpose(1, 2)
        gram = torch.bmm(feat, feat_t) / (ch * h * w)
        return gram


    def get_current_errors(self, has_GT, has_attr):

        if has_GT == True and has_attr == True:
            loss_dict = OrderedDict([('g_mskd_fake', self._loss_g_synthesis_fake.item()),
                                     ('g_m_mean', self._loss_g_mask.item()),
                                     ('g_m_smooth', self._loss_g_mask_smooth.item()),
                                     # ('g_m_hash', self._loss_g_mask_hash.item()),
                                     ('g_generate_face_smooth', self._loss_g_synth_smooth.item()),
                                     ('g_attr', self._loss_g_attr),
                                     ('g_generate_face_vaild', self._loss_g_vaild.item()),
                                     ('g_generate_face_hole', self._loss_g_hole.item()),
                                     ('g_generate_face_perceptual', self._loss_g_perceptual.item()),
                                     ('g_generate_face_style', self._loss_g_style.item()),
                                     ('d_real', self._loss_d_real.item()),
                                     ('d_fake', self._loss_d_fake.item()),
                                     ('d_gp', self._loss_d_gp.item()),
                                     ('d_attr', self._loss_d_attr)
                                     ])
        elif has_GT == False and has_attr == True:
            loss_dict = OrderedDict([('g_mskd_fake', self._loss_g_synthesis_fake.item()),
                         ('g_m_mean', self._loss_g_mask.item()),
                         ('g_m_smooth', self._loss_g_mask_smooth.item()),
                         # ('g_m_hash', self._loss_g_mask_hash.item()),
                         ('g_generate_face_smooth', self._loss_g_synth_smooth.item()),
                         ('g_attr', self._loss_g_attr),
                         ('d_real', self._loss_d_real.item()),
                         ('d_fake', self._loss_d_fake.item()),
                         ('d_gp', self._loss_d_gp.item()),
                         ('d_attr', self._loss_d_attr)
                         ])

        elif has_GT == False and has_attr == False:
            loss_dict = OrderedDict([('g_mskd_fake', self._loss_g_synthesis_fake.item()),
                         ('g_m_mean', self._loss_g_mask.item()),
                         ('g_m_smooth', self._loss_g_mask_smooth.item()),
                         # ('g_m_hash', self._loss_g_mask_hash.item()),
                         ('g_generate_face_smooth', self._loss_g_synth_smooth.item()),
                         ('d_real', self._loss_d_real.item()),
                         ('d_fake', self._loss_d_fake.item()),
                         ('d_gp', self._loss_d_gp.item()),
                         ])
        else:
            raise NotImplementedError('Not existing has_GT = False and has_attr = True')

        return loss_dict

    def get_current_scalars(self):

        return OrderedDict([('lr_G', self._current_lr_G), ('lr_D', self._current_lr_D)])

    def get_current_visuals(self):

        visuals = OrderedDict()
        visuals['1_batch_occ_img'] = self._vis_batch_occ_img
        visuals['2_batch_fake_img'] = self._vis_batch_fake_img
        visuals['3_batch_fake_img_mask'] = self._vis_batch_fake_img_mask
        visuals['4_batch_fake_img_synthesis'] = self._vis_batch_fake_synthesis
        visuals['5_batch_none_occ_img'] = self._vis_batch_none_occ_img

        return visuals

    def save(self, label):

        self._save_network(self._G, 'G', label)
        self._save_network(self._D, 'D', label)

        self._save_optimizer(self._optimizer_G, 'G', label)
        self._save_optimizer(self._optimizer_D, 'D', label)

    def load(self):

        load_epoch = self._opt.load_epoch

        self._load_network(self._G, 'G', load_epoch)

        if self._is_train:

            self._load_network(self._D, 'D', load_epoch)
            self._load_optimizer(self._optimizer_G, 'G', load_epoch)
            self._load_optimizer(self._optimizer_D, 'D', load_epoch)


    def update_learning_rate(self):

        lr_decay_G = self._opt.lr_G / self._opt.nepochs_decay
        self._current_lr_G -= lr_decay_G
        for param_group in self._optimizer_G.param_groups:
            param_group['lr'] = self._current_lr_G
        print('update G learning rate: %f -> %f' %  (self._current_lr_G + lr_decay_G, self._current_lr_G))

        lr_decay_D = self._opt.lr_D / self._opt.nepochs_decay
        self._current_lr_D -= lr_decay_D
        for param_group in self._optimizer_D.param_groups:
            param_group['lr'] = self._current_lr_D
        print('update D learning rate: %f -> %f' %  (self._current_lr_D + lr_decay_D, self._current_lr_D))
