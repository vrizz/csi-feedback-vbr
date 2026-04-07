from compressai.models import FactorizedPrior
from compressai.entropy_models import EntropyBottleneckVbr
from compressai.layers import GDN
import torch
import torch.nn as nn

from compressai.models.utils import conv, deconv


class CSIFactorizedPriorVbr(FactorizedPrior):
    def __init__(self, N, M, **kwargs):
        super().__init__(N, M, **kwargs)

        self.entropy_bottleneck = EntropyBottleneckVbr(self.M)

        self.g_a = nn.Sequential(
            conv(2, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 2),
        )

    def aux_loss(self):
        return sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneckVbr))

    def update(self, force=False, bin_width=None):
        if bin_width is None:
            rv = self.entropy_bottleneck.update(force=force)
        else:
            rv = self.entropy_bottleneck.update_variable(force=force, qs=bin_width)
        return rv

    def compress(self, x, bin_width=1.0):
        y = self.g_a(x)
        qs = torch.tensor(bin_width, device=x.device)
        y_strings = self.entropy_bottleneck.compress(y, qs=qs)

        return {"strings": [y_strings], "shape": y.size()[-2:]}

    def decompress(self, strings, shape, bin_width=1.0):
        assert isinstance(strings, list) and len(strings) == 1
        qs = torch.tensor(bin_width, device=self.entropy_bottleneck.cdf_length.device)
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape, qs=qs)
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}