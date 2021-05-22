import torch.nn as nn
import functools

class ModuleFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(module_name, *args, **kwargs):

        if module_name == 'generator_wgan':
            from .generator_wgan import Generator
            network = Generator(*args, **kwargs)
        elif module_name == 'discriminator_wgan_cls':
            from .discriminator_wgan_cls import Discriminator
            network = Discriminator(*args, **kwargs)
        else:
            raise ValueError("Module %s not recognized." % module_name)

        print("Module %s was created" % module_name)

        return network


class BaseModule(nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()
        self._name = 'BaseModule'

    @property
    def name(self):
        return self._name

    def init_weights(self):
        self.apply(self._weights_init_fn)

    def _weights_init_fn(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0)
        elif classname.find('BatchNorm2d') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def _get_norm_layer(self, norm_type='batch'):
        if norm_type == 'batch':
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        elif norm_type == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        elif norm_type =='batchnorm2d':
            norm_layer = nn.BatchNorm2d
        else:
            raise NotImplementedError('normalization layer [%s] is not found' % norm_type)

        return norm_layer
