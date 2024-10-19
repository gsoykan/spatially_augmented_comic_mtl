from torch import nn
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor


class ClassificationHead(nn.Module):
    def __init__(self,
                 emb_size: int = 512,
                 out_size: int = 1000,
                 use_mean_of_outputs_instead_of_cls: bool = False):
        """
        This uses all outputs of transformer encoder for classification,

        @param emb_size: transformer encoder embedding size
        @param n_classes: number of output classes
        @param use_mean_of_outputs_instead_of_cls: uses mean of all transformer encoder outputs
        instead of only cls output.
        """
        super().__init__()
        self.use_mean_of_outputs_instead_of_cls = use_mean_of_outputs_instead_of_cls
        if use_mean_of_outputs_instead_of_cls:
            self.model = nn.Sequential(
                # average over n, avg pooling
                Reduce('l b e -> b e', reduction='mean'),
                nn.LayerNorm(emb_size),
                nn.Linear(emb_size, out_size)
            )
        else:
            self.model = nn.Sequential(
                nn.LayerNorm(emb_size),
                nn.Linear(emb_size, out_size)
            )

    def forward(self, x: Tensor):
        if self.use_mean_of_outputs_instead_of_cls:
            return self.model(x)
        else:
            cls_outputs = x[0, :, :]
            return self.model(cls_outputs)
