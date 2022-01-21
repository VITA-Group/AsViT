import os, sys, time, argparse
from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from tqdm import tqdm
import numpy as np
import torch
from procedures import TEG
from procedures import prepare_seed, prepare_logger
from models import CellStructure, Transformer, count_matmul, matmul
from datasets import get_imagenet_dataset
from thop_modified import profile
from typing import List


SCALE_DEPTH = 1
SCALE_WIDTH = 4
PARAM_LIM = (80, 110) # million

HIDDEN_DIM = 8

# ascending False, descending True
REWARD_SORT_ORDER = {
    'ntk': False,
    'exp': True,
}

# round precision
TYPE2ROUND = {
    'ntk': 2,
    'exp': 5,
}


def grow_depth_choices(current, target_depth, all_choices, start_idx=0):
    if sum(current) == target_depth:
        all_choices.append(tuple(current))
        return
    for _idx in range(start_idx, len(current)):
        _current = list(current); _current[_idx] += 1
        grow_depth_choices(_current, target_depth, all_choices, start_idx=_idx+1)
    return


def merge_choices(grow_depth_choices, grow_width_choices):
    if len(grow_depth_choices) == 0:
        return list(grow_width_choices)
    if len(grow_width_choices) == 0:
        return list(grow_depth_choices)
    grow_choices = []
    for depth_choice in grow_depth_choices:
        for width_choice in grow_width_choices:
            grow_choices.append((depth_choice[0], width_choice[1]))
    return grow_choices


def grow_step(args, te_reward_generator, image_size, depth, hidden_dim, delta_depth=None, delta_dim=None, num_classes=1000, param_lim=100):
    assert not (delta_depth is None and delta_dim is None)
    reward_types = list(te_reward_generator._reward_types)
    reward2choice = {}
    for _type in reward_types:
        reward2choice[_type] = [] # list of (reward_value, choice)
    genotype = CellStructure(args.arch)
    depth_choices = []
    if delta_depth is not None:
        _depth_choices = []
        for _delta_depth in delta_depth:
            grow_depth_choices(list(depth), sum(depth)+_delta_depth, _depth_choices) # accumulate choices into _depth_choices
        for _choice in _depth_choices:
            depth_choices.append((_choice, hidden_dim))
    width_choices = []
    if delta_dim is not None:
        for _dim in delta_dim:
            if isinstance(delta_dim[0], float):
                # power grow
                if len(width_choices) > 0 and round(hidden_dim * (1 + _dim)) == width_choices[-1][1]: continue # duplicated choice
                width_choices.append((tuple(depth), round(hidden_dim * (1 + _dim))))
            else:
                # linear grow
                width_choices.append((tuple(depth), hidden_dim+_dim))
    grow_choices = merge_choices(depth_choices, width_choices)
    # print(grow_choices)
    choice2params = {}
    choice2flops = {}
    choice2reward = {} # choice: {reward_type: reward_value}
    choice2remove = []
    pbar = tqdm(grow_choices, position=0, leave=True)
    for choice in pbar:
        depth_next, dim_next = choice
        network = Transformer(img_size=image_size, patch_sizes=genotype.patch_sizes, stride=4, in_chans=3, num_classes=num_classes,
                      embed_dim=dim_next*SCALE_WIDTH, depths=np.array(depth_next)*SCALE_DEPTH, num_heads=genotype.heads,
                      window_sizes=genotype.window_sizes, mlp_ratios=genotype.mlp_ratios,
                      drop_rate=0., attn_drop_rate=0., drop_path_rate=0., use_checkpoint=False)
        flops, params = profile(network, inputs=(torch.randn(1, 3, image_size, image_size),), custom_ops={matmul: count_matmul}, verbose=False)
        params = sum(p.numel() for p in network.parameters() if p.requires_grad)
        flops /= 10**9
        params /= 10**6
        if params > param_lim:
            choice2remove.append(choice)
            continue
        network = Transformer(img_size=image_size, patch_sizes=genotype.patch_sizes, stride=4, in_chans=3, num_classes=num_classes,
                      embed_dim=dim_next, depths=np.array(depth_next), num_heads=genotype.heads,
                      window_sizes=genotype.window_sizes, mlp_ratios=genotype.mlp_ratios,
                      drop_rate=0., attn_drop_rate=0., drop_path_rate=0., use_checkpoint=False)
        te_reward_generator.set_network(network)
        choice2flops[choice] = flops
        choice2params[choice] = params
        results = dict(te_reward_generator.get_ntk_exp())
        for _type in reward_types:
            if choice not in choice2reward: choice2reward[choice] = {}
            choice2reward[choice][_type] = results[_type]
            reward2choice[_type].append([results[_type], choice])
        #############################
        torch.cuda.empty_cache()
        description = "Params %.2f | FLOPs %.2f"%(params, flops)
        if 'ntk' in results: description += " | NTK %.2f"%results['ntk']
        if 'exp' in results:
            description += " | Exp %.4f"%results['exp']
        pbar.set_description(description)
    for choice in choice2remove:
        grow_choices.remove(choice)
    if len(choice2params) == 0:
        return False, None, params, None
    rankings = {choice: [] for choice in grow_choices}  # dict of choice: [rank1, rank2, ...]
    for _type in reward_types:
        reward2choice[_type] = sorted(reward2choice[_type], key=lambda tup: tup[0], reverse=REWARD_SORT_ORDER[_type])  # ascending: we want choice to minimize ntk
        # print(_type, [(round(v, TYPE2ROUND[_type]), c) for v, c in reward2choice[_type]])
        for idx, (reward_value, choice) in enumerate(reward2choice[_type]):
            if idx == 0:
                rankings[choice].append(idx)
            else:
                if reward_value == reward2choice[_type][idx-1][0]:
                    # same reward_value as previous
                    rankings[choice].append(rankings[reward2choice[_type][idx-1][1]][-1]) # reuse the newly appended ranking of previous choice
                else:
                    rankings[choice].append(rankings[reward2choice[_type][idx-1][1]][-1] + 1) # reuse the newly appended ranking of previous choice
    rankings_list = [[k, v] for k, v in rankings.items()]  # list of [choice, [rank1, rank2, ...]]
    # ascending by sum of two rankings
    rankings_sum = sorted(rankings_list, key=lambda tup: sum(tup[1]), reverse=False)  # ascending list of [choice, [rank1, rank2, ...]]
    if len(rankings_sum) > 1 and sum(rankings_sum[0][1]) == sum(rankings_sum[1][1]):
        # there is a break even: follow criterion that has larger changing range
        reward_type2range = []
        for _type in reward_types:
            _range = (max([item[0] for item in reward2choice[_type]]) - min([item[0] for item in reward2choice[_type]])) / max([item[0] for item in reward2choice[_type]])
            reward_type2range.append([_type, _range])
        reward_type2range = sorted(reward_type2range, key=lambda tup: tup[1], reverse=True)  # descending order
        target_reward_type = reward_type2range[0][0] # choose type with largest changing range
        target_reward_list = [choice2reward[rankings_sum[0][0]][target_reward_type]]
        for _idx in range(1, len(rankings_sum)):
            if sum(rankings_sum[_idx-1][1]) == sum(rankings_sum[_idx][1]):
                target_reward_list.append(choice2reward[rankings_sum[_idx][0]][target_reward_type])
            break
        best_idx = np.argmin(target_reward_list)
        best_choice = rankings_sum[best_idx][0]
    else:
        best_choice = rankings_sum[0][0]
    return best_choice, choice2flops[best_choice], choice2params[best_choice], choice2reward[best_choice]


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
    root_dir = args.save_dir

    if args.dataset == 'imagenet':
        image_size = 224
    elif args.dataset == 'imagenet_64':
        image_size = 64
    else:
        raise NotImplementedError
    dataset_train, loader_train, dataset_eval, loader_eval = get_imagenet_dataset(data_path=args.data_path, no_aug=True, img_size=image_size, batch_size=args.batch_size, workers=0)

    results_summary = {}

    seed = args.rand_seed

    _size_curve = 10
    te_reward_generator = TEG(loader_train, loader_eval, size_curve=(_size_curve, 3, image_size, image_size), repeat=args.repeat,
                              reward_types=args.reward_types, buffer_size=args.te_buffer_size, batch_curve=6, constraint_weight=0)
        # "/%s-buffer%d-batch%d-hidden%d.scale%d-repeat%d"%('.'.join(args.reward_types), args.te_buffer_size, args.batch_size, HIDDEN_DIM, SCALE_WIDTH, args.repeat) + \
    args.save_dir = root_dir + \
        "/%s-batch%d-repeat%d"%('.'.join(args.reward_types), args.batch_size, args.repeat) + \
        "/{:}/seed{:}".format(args.timestamp, seed)
    args.rand_seed = seed
    logger = prepare_logger(args)
    print("<< ============== JOB (PID = %d) %s ============== >>"%(PID, '/'.join(args.save_dir.split("/")[-5:])))

    results_summary[seed] = []
    # starting from base arch
    hidden_dim = HIDDEN_DIM
    depth = [1, 1, 1, 1]
    # logger.log("<=================== seed %d =====================>"%(seed))
    prepare_seed(seed)
    # renew TE input
    te_reward_generator = TEG(loader_train, loader_eval, size_curve=(_size_curve, 3, image_size, image_size), repeat=args.repeat, reward_types=args.reward_types, buffer_size=args.te_buffer_size)

    while True:
        _grow_width_ratios = [0.05, 0.1, 0.15, 0.2]
        grow_next, flops, params, rewards = grow_step(args, te_reward_generator, image_size, depth, hidden_dim, delta_depth=[1], delta_dim=_grow_width_ratios, num_classes=1000, param_lim=PARAM_LIM[1])

        if grow_next: depth, hidden_dim = grow_next
        else: break
        results_summary[seed].append([(depth, hidden_dim), flops, params, rewards])
        genotype = CellStructure(args.arch); genotype.hidden_size = hidden_dim * SCALE_WIDTH; genotype.num_layers = list(depth)
        logger.log('step [{:3d}] grow: {:} | flops {:.2f} | params {:.2f} | ntk {:.2f} | exp {:.5f}'.format(
            len(results_summary[seed]), genotype.tostr(), flops, params, rewards['ntk'] if 'ntk' in rewards else -1, rewards['exp'] if 'exp' in rewards else -1))
        np.save(os.path.join(args.save_dir, "../results_summary.npy"), results_summary)
        if params >= PARAM_LIM[0]: break

    logger.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser("Grow")
    parser.add_argument('--data_path',          type=str,   help='Path to dataset')
    parser.add_argument('--dataset',            type=str,   default='imagenet', help='Choose between imagenet and imagenet_64.')
    parser.add_argument('--arch',           type=str,   help='genotype.')
    parser.add_argument('--workers',            type=int,   default=4,    help='number of data loading workers (default: 2)')
    parser.add_argument('--save_dir',           type=str,   help='Folder to save checkpoints and log.')
    parser.add_argument('--rand_seed',          type=int,   default=27,   help='manual seed')
    parser.add_argument('--timestamp', default='none', type=str, help='timestamp for logging naming')
    parser.add_argument('--batch_size',            type=int,   default=16,    help='batch size for ntk')
    parser.add_argument('--repeat',          type=int,   default=5,   help='repeat calculation for TEG')
    parser.add_argument('--te_buffer_size',        type=int,   default=20,   help='buffer size for TE reward generator')
    parser.add_argument('--reward_types',       type=str, default='ntk_exp',  help='type of supernet: basic or nasnet-super')
    args = parser.parse_args()
    main(args)
