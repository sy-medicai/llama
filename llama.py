class LoRALayer():
    def __init__(self, r: int, lora_alpha: int, lora_dropout: float, merge_weights: bool):
        self.r = r
        self.lora_alpha = lora_alpha
        if lora_dropout > 0.:
            self.lora_dropout == nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x : x
        self.merged = False
        self.merge_weights = merge_weights

class Embedding(nn.Embedding, LoRALayer)
    def __init__(self, num_embeddings: int, Embedding_dim: int, r: int = 0, lora_alpha: int = 1, merge_weights: bool = True, **kwargs):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
         LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0, merge_weights=merge_weights)
    if r > 0:
        self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
        self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
        self.scaling = self.lora_alpha / self.r
        self.weight.requires_grad = False

class Linear(nn.Linear, LoRALayer)
    def __init__(self, in_features: int, out_features: int, r : int=0, lora_alpha: int = 1, lora_dropout: float = 0., fan_in_fan_out: bool = False, merge_weights: bool = True, **kwargs):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False

class MergedLinear(nn.Linear, LoRALayer)
    def __init__(self, in_features: int, out_features: int, r: int = 0, lora_alpha: int = 1, lora_dropout: float = 0., enable_lora: List[bool], [False], fan_in_fan_out: bool = False, merge_weights: bool = True, **kwargs):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert out_features % len(enable_lora) ==0, 'enable_lora 길이는 out_features와 나누어 떨어져야 합니다'
        self.enable_lora = enable_lora
