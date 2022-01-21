import os, sys, time, argparse
import numpy as np
from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from tqdm import tqdm
import torch
from torch import nn
from torch.distributions import Categorical, Distribution
from procedures import TEG
from procedures   import prepare_seed, prepare_logger
from models import CellStructure, Transformer, count_matmul, matmul, SEARCH_SPACE
from datasets import get_imagenet_dataset
from thop_modified import profile
from typing import List


# https://github.com/pytorch/pytorch/issues/43250
class MultiCategorical(Distribution):

    def __init__(self, dists: List[Categorical]):
        super().__init__()
        self.dists = dists

    def log_prob(self, value):
        ans = []
        for d, v in zip(self.dists, torch.split(value, 1, dim=-1)):
            ans.append(d.log_prob(v.squeeze(-1)))
        return torch.stack(ans, dim=-1).sum(dim=-1)

    def entropy(self):
        return torch.stack([d.entropy() for d in self.dists], dim=-1).sum(dim=-1)

    def sample(self, sample_shape=torch.Size()):
        return torch.stack([d.sample(sample_shape) for d in self.dists], dim=-1)

    def derive(self):
        return torch.stack([dist.probs.argmax() for dist in self.dists], dim=0)


def multi_categorical_maker(nvec):
    def get_multi_categorical(logits):
        start = 0
        ans = []
        for n in nvec:
            ans.append(Categorical(logits=logits[start: start + n]))
            start += n
        return MultiCategorical(ans)
    return get_multi_categorical



class Policy(nn.Module):

    def __init__(self, search_space=SEARCH_SPACE):
        # search space: list of int, each represents #actions per dimention
        super(Policy, self).__init__()
        self.search_space = search_space
        self.arch_parameters = nn.ParameterList()
        self.space_dims = []
        self.space_keys = list(search_space.keys())
        for value in search_space.values():
            self.arch_parameters.append(nn.Parameter(1e-3*torch.randn(len(value))))
            self.space_dims.append(len(value))
        self.dist_maker = multi_categorical_maker(self.space_dims)

    def load_arch_params(self, arch_params):
        self.arch_parameters.data.copy_(arch_params)

    def action2arch_str(self, actions):
        _keyactions = {}
        for _idx, (key, value) in enumerate(self.search_space.items()):
            _keyactions[key] = value[actions[_idx].item()]
        arch_str = "{KERNEL_CHOICE1:d},{WINDOW_CHOICE1:d},{FFN_EXP_CHOICE1:d}|{KERNEL_CHOICE2:d},{WINDOW_CHOICE2:d},{FFN_EXP_CHOICE2:d}|{KERNEL_CHOICE3:d},{WINDOW_CHOICE3:d},{FFN_EXP_CHOICE3:d}|{KERNEL_CHOICE4:d},1,{FFN_EXP_CHOICE4:d}|{HEAD_CHOICE:d}".format(**_keyactions)
        return arch_str

    def generate_arch(self, arch, image_size, hidden_dim, depth, num_classes=1000, dropout=0, emb_dropout=0):
        if type(arch) in [list, tuple, torch.Tensor]:
            arch_str = self.action2arch_str(arch)
        elif isinstance(arch, str):
            arch_str = arch
        else:
            raise NotImplementedError
        genotype = CellStructure(arch_str)
        return arch_str, Transformer(img_size=image_size, patch_sizes=genotype.patch_sizes, stride=4, in_chans=3, num_classes=num_classes,
                             embed_dim=hidden_dim, depths=depth, num_heads=genotype.heads,
                             window_sizes=genotype.window_sizes, mlp_ratios=genotype.mlp_ratios,
                             drop_rate=0., attn_drop_rate=0., drop_path_rate=0., use_checkpoint=False)

    def genotype(self):
        self.distribution = self.dist_maker(logits=torch.cat([param for param in self.arch_parameters]))
        genotypes = self.distribution.derive() # ~ tensor of actions
        return self.action2arch_str(genotypes)

    def sample(self):
        self.distribution = self.dist_maker(logits=torch.cat([param for param in self.arch_parameters]))
        actions = self.distribution.sample()
        log_prob = self.distribution.log_prob(actions)
        return actions, log_prob


class ExponentialMovingAverage(object):
    """Class that maintains an exponential moving average."""
    def __init__(self, momentum):
        self._numerator   = 0
        self._denominator = 0
        self._momentum    = momentum

    def update(self, value):
        self._numerator = self._momentum * self._numerator + (1 - self._momentum) * value
        self._denominator = self._momentum * self._denominator + (1 - self._momentum)

    def value(self):
        """Return the current value of the moving average"""
        return self._numerator / self._denominator


def main(args):
    PID = os.getpid()
    if args.timestamp == 'none':
        args.timestamp = "{:}".format(time.strftime('%h-%d-%C_%H-%M-%s', time.gmtime(time.time())))

    assert torch.cuda.is_available(), 'CUDA is not available.'
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads( args.workers )

    args.reward_types = args.reward_types.split('_')
    args.save_dir = args.save_dir + \
        "/LR%.2f-steps%d-%s-buffer%d-batch%d-repeat%d"%(args.learning_rate, args.total_steps, '.'.join(args.reward_types), args.te_buffer_size, args.batch_size, args.repeat) + \
        "/{:}/seed{:}".format(args.timestamp, args.rand_seed)
    logger = prepare_logger(args)

    if args.dataset == 'imagenet':
        image_size = 224
    else:
        raise NotImplementedError
    logger.log("preparing dataset...")
    dataset_train, loader_train, dataset_eval, loader_eval = get_imagenet_dataset(data_path=args.data_path, no_aug=True, img_size=image_size,
                                                                                  batch_size=args.batch_size, workers=0)

    eps = np.finfo(np.float32).eps.item()
    logger.log('eps       : {:}'.format(eps))

    # REINFORCE
    trace = []
    total_steps = args.total_steps
    hidden_dim = 32
    depth = [1, 1, 1, 1]

    seed = args.rand_seed

    print("<< ============== JOB (PID = %d) %s ============== >>"%(PID, '/'.join(args.save_dir.split("/")[-5:])))
    _size_curve = 10
    te_reward_generator = TEG(loader_train, loader_eval, size_curve=(_size_curve, 3, image_size, image_size), repeat=args.repeat,
                              reward_types=args.reward_types, buffer_size=args.te_buffer_size, batch_curve=6)
    prepare_seed(seed)
    te_reward_generator.reset()
    policy = Policy().cuda()
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    baseline = ExponentialMovingAverage(args.EMA_momentum)
    arch_str_history = [] # save the arch to be derived by the NAS algorithm at anytime
    time_te_total = 0
    pbar = tqdm(range(total_steps), position=0, leave=True)
    for _step in pbar:
        # record history of policy and optimizer states

        action, log_prob = policy.sample()
        arch_str, network = policy.generate_arch(action, image_size, hidden_dim, depth)
        arch_str_history.append(arch_str)

        _start_time = time.time()
        flops, params = profile(network, inputs=(torch.randn(1, 3, image_size, image_size),), custom_ops={matmul: count_matmul}, verbose=False)
        params = sum(p.numel() for p in network.parameters() if p.requires_grad) # thop did not consider nn.Parameter
        logger.writer.add_scalar("TE/flops", flops, _step)
        logger.writer.add_scalar("TE/params", params, _step)
        reward = te_reward_generator.step(network)
        description = " | Params %.0f | FLOPs %.0f"%(params, flops)
        if 'ntk' in te_reward_generator._buffers:
            description += " | NTK %.2f"%te_reward_generator._buffers['ntk'][-1]
            logger.writer.add_scalar("TE/NTK", te_reward_generator._buffers['ntk'][-1], _step)
        if 'exp' in te_reward_generator._buffers:
            description += " | Exp %.4f"%te_reward_generator._buffers['exp'][-1]
            logger.writer.add_scalar("TE/Exp", te_reward_generator._buffers['exp'][-1], _step)
        time_te_total += (time.time() - _start_time)
        logger.writer.add_scalar("reinforce/entropy", policy.distribution.entropy(), _step)
        logger.writer.add_scalar("reward/reward", reward, _step)

        pbar.set_description("Entropy {entropy:.2f} | Reward {reward:.2f}".format(entropy=policy.distribution.entropy(), reward=reward) + description)

        trace.append((reward, arch_str))
        baseline.update(reward)
        # calculate loss
        policy_loss = ( -log_prob * (reward - baseline.value()) ).sum()
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

    arch_str_derived, network_derived = policy.generate_arch(policy.genotype(), image_size, hidden_dim, depth)
    flops, params = profile(network_derived, inputs=(torch.randn(1, 3, image_size, image_size),), custom_ops={matmul: count_matmul}, verbose=False)
    params = sum(p.numel() for p in network_derived.parameters() if p.requires_grad) # thop did not consider nn.Parameter
    te_reward_generator.set_network(network_derived.cuda())
    results = [arch_str_derived, flops, params, te_reward_generator.get_ntk(), te_reward_generator.get_curve_complexity()]

    logger.log('[Policy] {:s} | flops {:.0f} | params {:.0f} | ntk {:.2f} | exp {:.5f}'.format(*results))

    logger.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser("Reinforce")
    parser.add_argument('--data_path',          type=str,   help='Path to dataset')
    parser.add_argument('--dataset',            type=str,   default='imagenet', help='Choose between imagenet and imagenet_64.')
    parser.add_argument('--learning_rate',      type=float, default=0.08, help='The learning rate for REINFORCE.')
    parser.add_argument('--total_steps',      type=int, default=500, help='Number of iterations for REINFORCE.')
    parser.add_argument('--EMA_momentum',       type=float, default=0.9, help='The momentum value for EMA.')
    parser.add_argument('--workers',            type=int,   default=4,    help='number of data loading workers (default: 2)')
    parser.add_argument('--save_dir',           type=str,   help='Folder to save checkpoints and log.')
    parser.add_argument('--rand_seed',          type=int,   default=8,   help='manual seed')
    parser.add_argument('--timestamp', default='none', type=str, help='timestamp for logging naming')
    parser.add_argument('--batch_size',            type=int,   default=16,    help='batch size for ntk')
    parser.add_argument('--repeat',          type=int,   default=5,   help='repeat calculation for TEG')
    parser.add_argument('--te_buffer_size',        type=int,   default=20,   help='buffer size for TE reward generator')
    parser.add_argument('--reward_types',       type=str, default='ntk_exp',  help='type of supernet: basic or nasnet-super')
    args = parser.parse_args()
    main(args)
