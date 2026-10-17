import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor
from features import LIN, GB


class HybridRateModel:
    """log(rate / lane_distance) = Huber-linear(trend, log market index, log weight, quote level)
                                  + gradient-boosted residual (lane, equipment, distance, quote_signal ...)
    The linear layer extrapolates the time trend / market sensitivity into unseen months; trees cannot."""

    def __init__(self, seeds=(0,), max_iter=400, lr=0.05):
        self.seeds, self.max_iter, self.lr = seeds, max_iter, lr

    def fit(self, d, y):
        self.mu = d[LIN].values.mean(0)
        self.lin = HuberRegressor(max_iter=1000).fit(d[LIN].values - self.mu, y)
        r = y - self.lin.predict(d[LIN].values - self.mu)
        cat = [GB.index(c) for c in ("eq", "pickup_c", "delivery_c")]
        self.gbs = [HistGradientBoostingRegressor(
            loss="absolute_error", max_iter=self.max_iter, learning_rate=self.lr, max_leaf_nodes=31,
            min_samples_leaf=30, categorical_features=cat, random_state=s, early_stopping=False).fit(d[GB], r)
            for s in self.seeds]
        return self

    def predict_log_rpm(self, d):
        return self.lin.predict(d[LIN].values - self.mu) + np.mean([g.predict(d[GB]) for g in self.gbs], 0)

    def predict_rate(self, d):
        return np.exp(self.predict_log_rpm(d)) * d.lane_dist.values
