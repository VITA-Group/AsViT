from collections import OrderedDict


# genotype
class Structure:
    def __init__(self, genotype_str):
        # genotype: e.g., "8,2,3,5|4,1,2,2|4,1,4,5|4,1,6,2|32,180"
        assert isinstance(genotype_str, str)
        self.genotype_str = genotype_str
        genotype = genotype_str.split('|')
        self.num_stages = len(genotype[:-1])
        self.patch_sizes = [int(stage.split(',')[0]) for stage in genotype[:-1]]
        self.window_sizes = [int(stage.split(',')[1]) for stage in genotype[:-1]]
        self.mlp_ratios = [int(stage.split(',')[2]) for stage in genotype[:-1]]
        head = int(genotype[-1].split(",")[0])
        self.heads = [head//8, head//4, head//2, head]
        try:
            self.num_layers = [int(stage.split(',')[3]) for stage in genotype[:-1]]
        except Exception:
            self.num_layers = [1] * 4
        try:
            self.hidden_size = int(genotype[-1].split(",")[1])
        except Exception:
            self.hidden_size = 32

    def tostr(self):
        string = []
        for p, w, r, l in zip(self.patch_sizes, self.window_sizes, self.mlp_ratios, self.num_layers):
            string.append("%d,%d,%d,%d"%(p, w, r, l))
        string = '|'.join(string) + "|" + "%d,%d"%(self.heads[-1], self.hidden_size)
        self.genotype_str = string
        return string

    def __repr__(self):
        return ('{name}({node_info})'.format(name=self.__class__.__name__, node_info=self.tostr()))

    def __len__(self):
        return len(self.num_stages)