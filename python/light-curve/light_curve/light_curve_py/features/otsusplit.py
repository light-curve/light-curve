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

    - Depends on: **magnitude**
    - Minimum number of observations: **2**
    - Number of features: **4**

    Otsu, Nobuyuki 1979. [DOI:10.1109/tsmc.1979.4310076](https://doi.org/10.1109/tsmc.1979.4310076)
    """

    def _eval(self, t, m, sigma=None):
        n = len(m)
        amounts = np.arange(1, n)
        m = np.sorted(m)

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
        return "otsu_mean_diff", "otsu_std_lower", "otsu_std_upper", "otsu_up_to_all_ratio"

    @property
    def descriptions(self):
        return (
            "difference between mean values of Otsu split subsets",
            "standard deviation for observations below the threshold given by Otsu method",
            "standard deviation for observations above the threshold given by Otsu method",
            "ratio of quantity of observations above the threshold given by Otsu method to quantity of all observations",
        )

    @property
    def size(self):
        return 4


__all__ = ("OtsuSplit",)
