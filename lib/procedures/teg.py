import os
import copy
import numpy as np
import torch
from torch import nn
from torch.autograd import grad
from procedures import get_ntk_n


reward_type2index = {
    'ntk': 0,
    'exp': 1,
    'constraint': 3
}
index2reward_type = {
    0: 'ntk',
    1: 'exp',
    3: 'constraint'
}


def kaiming_normal_fanin_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)


def kaiming_normal_fanout_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)


def init_model(model, method='kaiming_norm_fanin'):
    if method == 'kaiming_norm_fanin':
        model.apply(kaiming_normal_fanin_init)
    elif method == 'kaiming_norm_fanout':
        model.apply(kaiming_normal_fanout_init)
    return model


class TEG(object):
    def __init__(self, loader, loader_val, class_num=1000, repeat=3, size_curve=(500, 3, 16, 16), batch_curve=6, reward_types=["ntk", "exp"], buffer_size=10, constraint_weight=0.):
        # self.__super__()
        self.repeat = repeat
        self.constraint_weight = constraint_weight # e.g. FLOPs constraint
        self.batch_size_curve = batch_curve

        self.reward_type2index = reward_type2index
        self.index2reward_type = index2reward_type
        self._reward_types = reward_types
        self._reward_sign = {"ntk": -1, "exp": 1, "constraint": -1} # ntk: lower the better; exp: higher the better
        self._buffers = {key: [] for key in self._reward_types}
        self._buffers['constraint'] = []
        self._buffers_bad = [] # indicator of bad architectures
        self._buffers_change = {key: [] for key in self._reward_types}
        self._buffers_change['constraint'] = []
        self._buffer_length = buffer_size
        self._class_num = class_num
        # build fixed data samples
        self._ntk_input_data = []
        for i, (inputs, targets) in enumerate(loader):
            if i >= self.repeat: break
            self._ntk_input_data.append((inputs, targets))
            self.batch_size = len(inputs)
        self._ntk_target_data = [] # for NTK kernel regression
        for i, (inputs, targets) in enumerate(loader_val):
            if i >= self.repeat: break
            self._ntk_target_data.append((inputs, targets))
        # Curve complexity
        n_interp, C, H, W = size_curve
        self.theta = []; self.curve_input = []
        for _ in range(self.repeat):
            self.theta.append(torch.linspace(0, 2 * np.pi, n_interp))
            self.theta[-1].requires_grad_(True)
            self.curve_input.append(torch.matmul(torch.svd(torch.randn(H*W*C, 2))[0], torch.stack([torch.cos(self.theta[-1]), torch.sin(self.theta[-1])])).T.reshape((n_interp, C, H, W)).cuda(non_blocking=True))
            self.curve_input[-1].requires_grad_(True)

    def reset(self, constraint_weight=0):
        self.constraint_weight = constraint_weight
        # self._reward_types = reward_types
        self._buffers = {key: [] for key in self._reward_types}; self._buffers['constraint'] = []
        self._buffers_bad = [] # indicator of bad architectures
        self._buffers_change = {key: [] for key in self._reward_types}; self._buffers_change['constraint'] = []

    def set_network(self, network):
        if hasattr(self, "_networks"):
            del self._networks
        self._networks = []
        for _ in range(self.repeat):
            net = copy.deepcopy(network)
            net.apply(net._init_weights)
            self._networks.append(net.cuda())

    def get_ntk(self):
        for net in self._networks:
            net.switch_norm('ln')
        ntks = get_ntk_n(self._ntk_input_data, self._networks, criterion=torch.nn.CrossEntropyLoss(), train_mode=True, num_batch=1, num_classes=self._class_num)
        for network in self._networks:
            network.zero_grad()
        torch.cuda.empty_cache()
        return np.mean(ntks)

    def get_curve_complexity(self):
        for net in self._networks:
            net.switch_norm('id')
        LE = [0 for _ in range(len(self._networks))]
        for net_idx, network in enumerate(self._networks):
            network = network.cuda()
            network.train()
            network.zero_grad()
            _idx = 0
            while _idx < len(self.curve_input[net_idx]):
                output = network.forward_features(self.curve_input[net_idx][_idx:_idx+self.batch_size_curve])[1]
                _idx += self.batch_size_curve
                output = output.reshape(output.size(0), -1)
                n, c = output.size()
                jacobs = []
                for coord in range(c):
                    output[:, coord].backward(torch.ones_like(output[:, coord]), retain_graph=True)
                    # actually only "batch_size" number of thetas have grad, but it is ok, since zeros won't contribute to gE.sum()
                    jacobs.append(self.theta[net_idx].grad.detach().clone())
                    self.theta[net_idx].grad.zero_()
                jacobs = torch.stack(jacobs, 0)
                jacobs = jacobs.permute(1, 0) # num_theta x c
                gE = torch.einsum('nd,nd->n', jacobs, jacobs).sqrt()
                LE[net_idx] += gE.sum().item()
                torch.cuda.empty_cache()
            network = network.cpu()
            torch.cuda.empty_cache()
        for net in self._networks:
            net.switch_norm('ln')
        return np.mean(LE)

    def get_curve_complexity_gauss(self):
        for net in self._networks:
            net.switch_norm('id')
        LG = [0 for _ in range(len(self._networks))]
        for net_idx, network in enumerate(self._networks):
            network = network.cuda()
            network.train()
            network.zero_grad()
            _idx = 0
            v_s = [] # 1st derivative
            while _idx < len(self.curve_input[net_idx]):
                output = network.forward_features(self.curve_input[net_idx][_idx:_idx+self.batch_size_curve])[1]
                output = output.reshape(output.size(0), -1)
                n, c = output.size()
                _v_s = [] # 1st derivative
                for coord in range(c):
                    v = grad(output[:, coord].sum(), self.theta[net_idx], create_graph=True, retain_graph=True)[0][_idx:_idx+self.batch_size_curve] # batch size (of thetas)
                    _v_s.append(v.detach().clone())
                v_s.append(torch.stack(_v_s, 0).permute(1, 0)) # bach_size x c
                _idx += self.batch_size_curve
            v_s = torch.cat(v_s, 0) # num_thetas x c
            v_s_norm = v_s.norm(2, dim=1, keepdim=True) # norm over c of all thetas
            _idx = 0
            while _idx < len(self.curve_input[net_idx]):
                output = network.forward_features(self.curve_input[net_idx][_idx:_idx+self.batch_size_curve])[1]
                output = output.reshape(output.size(0), -1)
                n, c = output.size()
                d_v_hat_s = [] # 2nd derivative
                for coord in range(c):
                    v = grad(output[:, coord].sum(), self.theta[net_idx], create_graph=True, retain_graph=True)[0][_idx:_idx+self.batch_size_curve] # batch size (of thetas)
                    d_v_hat = grad((v / v_s_norm[_idx:_idx+self.batch_size_curve]).sum(), self.theta[net_idx], create_graph=True, retain_graph=True)[0][_idx:_idx+self.batch_size_curve] # batch size (of thetas)
                    d_v_hat_s.append(d_v_hat.detach().clone())
                    del v
                d_v_hat_s = torch.stack(d_v_hat_s, 0).permute(1, 0) # batch_size_curve x c
                gG = torch.einsum('nd,nd->n', d_v_hat_s, d_v_hat_s).sqrt()
                LG[net_idx] += gG.sum().item()
                torch.cuda.empty_cache()
                _idx += self.batch_size_curve
            network = network.cpu()
            torch.cuda.empty_cache()
        return np.mean(LG)

    def get_extrinsic_curvature(self):
        for net in self._networks:
            net.switch_norm('id')
        kappa = [0 for _ in range(len(self._networks))]
        for net_idx, network in enumerate(self._networks):
            network = network.cuda()
            network.train()
            network.zero_grad()
            _idx = 0
            while _idx < len(self.curve_input[net_idx]):
                output = network.forward_features(self.curve_input[net_idx][_idx:_idx+self.batch_size_curve])[1]
                output = output.reshape(output.size(0), -1)
                n, c = output.size()
                v_s = [] # 1st derivative
                a_s = [] # 2nd derivative
                for coord in range(c):
                    v = grad(output[:, coord].sum(), self.theta[net_idx], create_graph=True, retain_graph=True)[0][_idx:_idx+self.batch_size_curve] # batch size (of thetas)
                    a = grad(v.sum(), self.theta[net_idx], create_graph=True, retain_graph=True)[0][_idx:_idx+self.batch_size_curve] # batch size (of thetas)
                    v_s.append(v.detach().clone())
                    a_s.append(a.detach().clone())
                v_s = torch.stack(v_s, 0).permute(1, 0) # batch_size_curve x c
                a_s = torch.stack(a_s, 0).permute(1, 0) # batch_size_curve x c
                vv = torch.einsum('nd,nd->n', v_s, v_s)
                aa = torch.einsum('nd,nd->n', a_s, a_s)
                va = torch.einsum('nd,nd->n', v_s, a_s)
                kappa[net_idx] += (vv**(-3/2) * (vv * aa - va ** 2).sqrt()).sum().item()
                torch.cuda.empty_cache()
                _idx += self.batch_size_curve
            network = network.cpu()
            torch.cuda.empty_cache()
        return np.mean(kappa)

    def _update_bad_cases(self, reward_type, reward):
        # re-set "reward_type" of bad architectures to "reward"
        for _type in self._reward_types:
            for _idx, isbad in enumerate(self._buffers_bad):
                if isbad:
                    self._buffers[_type][_idx] = reward
            for _idx, isbad in enumerate(self._buffers_bad):
                if isbad:
                    self._buffers_change[_type][_idx] = (self._buffers[_type][_idx] - self._buffers[_type][_idx-1]) / (max(self._buffers[_type][max(0, _idx+1-self._buffer_length):_idx+1]) - min(self._buffers[_type][max(0, _idx+1-self._buffer_length):_idx+1]) + 1e-6)
                    if _idx + 1 < len(self._buffers_bad):
                        self._buffers_change[_type][_idx+1] = (self._buffers[_type][_idx+1] - self._buffers[_type][_idx]) / (max(self._buffers[_type][max(0, _idx+2-self._buffer_length):_idx+2]) - min(self._buffers[_type][max(0, _idx+2-self._buffer_length):_idx+2]) + 1e-6)

    def get_reward(self):
        #  changing range comparison ######
        _reward = _type = 0
        if len(self._buffers[self._reward_types[0]]) <= 1:
            # dummy reward for step 0
            return 0
        type_reward = [] # tuples of (type, reward)
        for _type in self._reward_types:
            var = self._buffers_change[_type][-1]
            type_reward.append((self.reward_type2index[_type], self._reward_sign[_type] * var))
        if 'constraint' in self._buffers and len(self._buffers['constraint']) > 0:
            var = self._buffers_change['constraint'][-1]
            type_reward.append((self.reward_type2index['constraint'], self._reward_sign['constraint'] * var * self.constraint_weight))
        if len(type_reward) > 0:
            _reward = sum([_r for _t, _r in type_reward])
        return _reward

    def _buffer_insert(self, results):
        if len(self._buffers[self._reward_types[0]]) == 0:
            self._buffers_bad.append(results['bad'])
            for _type in self._reward_types:
                self._buffers_change[_type].append(0)
                self._buffers[_type].append(results[_type])
            if 'constraint' in results:
                self._buffers_change['constraint'].append(0)
                self._buffers['constraint'].append(results['constraint'])
        else:
            if results['bad']:
                # set ntk of bad architecture as worst case in current buffer
                if 'ntk' in self._reward_types: results['ntk'] = max(self._buffers['ntk'])
            else:
                if 'ntk' in self._reward_types and results['ntk'] > max(self._buffers['ntk']):
                    self._update_bad_cases('ntk', results['ntk'])
            self._buffers_bad.append(results['bad'])
            for _type in self._reward_types:
                self._buffers[_type].append(results[_type])
                var = (self._buffers[_type][-1] - self._buffers[_type][-2]) / (max(self._buffers[_type][-self._buffer_length:]) - min(self._buffers[_type][-self._buffer_length:]) + 1e-6)
                self._buffers_change[_type].append(var)
            if 'constraint' in results:
                self._buffers['constraint'].append(results['constraint'])
                var = (self._buffers['constraint'][-1] - self._buffers['constraint'][-2]) / (max(self._buffers['constraint'][-self._buffer_length:]) - min(self._buffers['constraint'][-self._buffer_length:]) + 1e-6)
                self._buffers_change['constraint'].append(var)

    def get_ntk_exp(self):
        results = {}
        if 'ntk' in self._reward_types:
            ntk = self.get_ntk()
            results['ntk'] = ntk
            results['bad'] = ntk==-1 # networks of bad gradients
        if 'exp' in self._reward_types:
            exp = self.get_curve_complexity()
            results['exp'] = exp
            results['bad'] = False # networks of bad gradients
        torch.cuda.empty_cache()
        return results

    def step(self, network, constraint=None, verbose=False):
        self.set_network(network)
        results = self.get_ntk_exp()
        if constraint is not None:
            results['constraint'] = constraint
        self._buffer_insert(results)
        if verbose:
            print("NTK buffer:", self._buffers['ntk'][-self._buffer_length:])
            print("NTK change buffer:", self._buffers_change['ntk'][-self._buffer_length:])
            print("Exp buffer:", self._buffers['exp'][-self._buffer_length:])
            print("Exp change buffer:", self._buffers_change['exp'][-self._buffer_length:])
            if constraint is not None:
                print("Constraint buffer:", self._buffers['constraint'][-self._buffer_length:])
                print("Constraint change buffer:", self._buffers_change['constraint'][-self._buffer_length:])
        reward = self.get_reward()
        # reward larger the better
        return reward

    def _buffer_rank_best(self):
        # return the index of the best based on rankings over three buffers
        rankings = {}
        buffers_sorted = {}
        rankings_all = []
        for _type in self._reward_types:
            buffers_sorted[_type] = sorted(self._buffers[_type], reverse=self._reward_sign[_type]==1) # by default ascending
            num_samples = len(buffers_sorted[_type])
            rankings[_type] = [ buffers_sorted[_type].index(value) for value in self._buffers[_type] ]
        for _idx in range(num_samples):
            rankings_all.append(sum([ rankings[_type][_idx] for _type in rankings.keys() ]))
        return np.argmin(rankings_all)
