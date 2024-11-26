import pandas as pd

class BaseStats:

    # class attributes
    fields = []
    has_time_dim = False
    has_multi_dim = False

    # @classmethod
    # def for_single_overall(cls, x: pd.Series): pass

    # @classmethod
    # def for_single_periodic(cls, x: pd.Series, period): pass  # would be a DF (axis=0 is time dim)

    # @classmethod
    # def for_single_rolling(cls, x: pd.Series, lookback): pass  # similarly to periodic


class BasicStats(BaseStats):    
    # min/max/mean/med/std/skew/kurt
    fields = ["min", "max", "mean", "med", "std", "skew", "kurt"]
    

    @classmethod
    def for_single_overall(cls, x: pd.Series): 
        # calculate all fields for corresponding stats, using pandas Series APIs 
        return pd.Series({
            "min": x.min(),
            "max": x.max(),
            "mean": x.mean(),
            "med": x.median(),
            "std": x.std(),
            "skew": x.skew(),
            "kurt": x.kurt()
        })

    @classmethod
    def for_single_periodic(cls, x: pd.Series, period): 
        # calculate all fields for corresponding stats, using pandas Series APIs
        return pd.DataFrame({
            "min": x.resample(period).min(),
            "max": x.resample(period).max(),
            "mean": x.resample(period).mean(),
            "med": x.resample(period).median(),
            "std": x.resample(period).std(),
            "skew": x.resample(period).skew(),
            "kurt": x.resample(period).kurt()
        })

    @classmethod
    def for_single_rolling(cls, x: pd.Series, lookback):
        # calculate all fields for corresponding stats, using pandas Series APIs
        return pd.DataFrame({
            "min": x.rolling(lookback).min(),
            "max": x.rolling(lookback).max(),
            "mean": x.rolling(lookback).mean(),
            "med": x.rolling(lookback).median(),
            "std": x.rolling(lookback).std(),
            "skew": x.rolling(lookback).skew(),
            "kurt": x.rolling(lookback).kurt()
        })

class Quantiles(BasicStats):
    fields = ["q01", "q05", "q10", "q25", "q50", "q75", "q90", "q95", "q99"]
    ...

class PairStats(BaseStats):
    fields = ["corr", "cov", "beta", "rankcorr"]
    ...


