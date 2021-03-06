from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        self._parser.add_argument('--n_threads_train', default=1, type=int, help='# threads for loading data')
        self._parser.add_argument('--num_iters_validate', default=1, type=int, help='# batches to use when validating')
        self._parser.add_argument('--print_freq_s', type=int, default=300, help='frequency of showing training results on console')
        self._parser.add_argument('--display_freq_s', type=int, default=600, help='frequency [s] of showing training results on screen')
        self._parser.add_argument('--save_latest_freq_s', type=int, default=3600, help='frequency of saving the latest results')
        self._parser.add_argument('--nepochs_no_decay', type=int, default=100, help='# of epochs at starting learning rate')
        self._parser.add_argument('--nepochs_decay', type=int, default=10, help='# of epochs to linearly decay learning rate to zero')
        self._parser.add_argument('--train_G_every_n_iterations', type=int, default=5, help='train G every n interations')

        self._parser.add_argument('--lr_G', type=float, default=0.0001, help='initial learning rate for G adam')
        self._parser.add_argument('--G_adam_b1', type=float, default=0.5, help='beta1 for G adam')
        self._parser.add_argument('--G_adam_b2', type=float, default=0.999, help='beta2 for G adam')
        self._parser.add_argument('--lr_D', type=float, default=0.0001, help='initial learning rate for D adam')
        self._parser.add_argument('--D_adam_b1', type=float, default=0.5, help='beta1 for D adam')
        self._parser.add_argument('--D_adam_b2', type=float, default=0.999, help='beta2 for D adam')
        
        self._parser.add_argument('--attr_nc', type=int, default=7, help='number of attr')

        self._parser.add_argument('--lambda_D_prob', type=float, default=1, help='lambda for real/fake discriminator loss')
        self._parser.add_argument('--lambda_D_gp', type=float, default=10, help='lambda gradient penalty loss')
        
        self._parser.add_argument('--lambda_D_attr', type=float, default=2000, help='lambda attr penalty loss')

        self._parser.add_argument('--lambda_mask', type=float, default=1, help='lambda mask loss')
        self._parser.add_argument('--lambda_mask_smooth', type=float, default=1e-5, help='lambda mask smooth loss')
        
        self._parser.add_argument('--lambda_g_style', type=float, default=120, help='')
        self._parser.add_argument('--lambda_g_perceptual', type=float, default=0.05, help='')
        self._parser.add_argument('--lambda_g_syhth_smooth', type=float, default=1e-5, help='')
        self._parser.add_argument('--lambda_g_hole', type=float, default=6, help='')
        self._parser.add_argument('--lambda_g_valid', type=float, default=1, help='')
        self._parser.add_argument('--lambda_g_hash', type=float, default=1, help='')

        self.is_train = False
