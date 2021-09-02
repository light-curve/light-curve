from dataclasses import dataclass
import numpy as np

from ._base import BaseFeature


@dataclass()
class OtsuSplit(BaseFeature):
    """Difference of subset means, standard deviation of the lower subset, standard deviation
    of the upper subset and upper-to-all observation count ratio for two subsets of magnitudes
    obtained by Otsu's method split.

    Otsu's method is used to perform automatic thresholding. The algorithm returns a single
    threshold that separate values into two classes. This threshold is determined by minimizing
    intra-class intensity variance, or equivalently, by maximizing inter-class variance.

    Otsu, Nobuyuki 1979. [DOI:10.1109/tsmc.1979.4310076]
    """

    def __call__(self, t, m, sigma=None, sorted=None, fill_value=None):
        n = len(m)
        amounts = np.arange(1, n)

        w0 = amounts / n
        w1 = 1 - w0

        cumsum0 = np.cumsum(m)[:-1]
        cumsum1 = np.cumsum(m[::-1])[:-1][::-1]
        mean0 = cumsum0 / amounts
        mean1 = cumsum1 / amounts[::-1]

        inter_class_variance = w0 * w1 * (mean0 - mean1) ** 2
        arg = np.argmax(inter_class_variance)

        std_lower = np.std(m[: arg + 1], ddof=1)
        std_upper = np.std(m[arg + 1 :], ddof=1)

        if len(m[: arg + 1]) == 1:
            std_lower = 0
        if len(m[arg + 1 :]) == 1:
            std_upper = 0

        up_to_all_ratio = (arg + 1) / n

        return mean1[arg] - mean0[arg], std_lower, std_upper, up_to_all_ratio

    @property
    def names(self):
        pass

    @property
    def descriptions(self):
        pass


__all__ = ("OtsuSplit",)
