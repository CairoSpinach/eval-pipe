from dataclasses import dataclass
from typing import List, Union
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass
class BinInfo:
    label: str
    binType: str  # 'int', 'array'
    bins: np.ndarray

def parse_bin(signal, bins=None, m=None):
    # basically to unify the return values from its most convenient form
    if bins is None:
        return []
    if isinstance(bins, np.ndarray):
        return [bins]
    elif isinstance(bins, int):
        if m is None:
            m = signal.abs().max()
        return [np.linspace(-m, m, bins + 1)]
    elif isinstance(bins, list):
        # note it's different from ndarray, assume it's list of ints
        # assert it's list of ints
        return [parse_bin(signal, i) for i in bins]


def _get_bin_mean(signal, ret, bins, statistic='mean', error_bars=True, confidence_level=0.95):
    binned = pd.cut(signal, bins=bins)
    grouped = ret.groupby(binned)
    
    # Base statistics
    stats_df = pd.DataFrame({
        'count': grouped.count(),
        statistic: grouped.agg(statistic)
    })
    
    # Add error bar calculations if requested and using mean
    if error_bars and statistic == 'mean':
        # Calculate std for each bin
        stats_df['std'] = grouped.std()
        
        # Standard error = std / sqrt(n)
        stats_df['stderr'] = stats_df['std'] / np.sqrt(stats_df['count'])
        
        # Calculate error margins using t-distribution
        from scipy import stats
        t_value = stats.t.ppf((1 + confidence_level) / 2, stats_df['count'] - 1)
        stats_df['error_margin'] = t_value * stats_df['stderr']
        
        # Calculate confidence intervals
        stats_df['ci_lower'] = stats_df[statistic] - stats_df['error_margin']
        stats_df['ci_upper'] = stats_df[statistic] + stats_df['error_margin']
    
    return stats_df


# def bin_mean_stats(signal, ret, bins=None, statistic='mean', m=None):
#     bins_list = parse_bin(signal, bins, m=m, )
#     result = []
#     for info in bins_list:
#         result.append((
#             info.label,
#             _get_bin_mean(signal, ret, info.bins, statistic=statistic)
#         ))
#     return result


# def quantile_mean_stats(signal, ret, quantile=None, statistic='mean', m=None):
#     # ist = parse_bin(signal, bins, m=m, )
#     result = []
#     for q in quantile:
#         quantiles = pd.qcut(signal, q=q)
    
#         # Group and compute statistics
#         grouped = ret.groupby(quantiles)
#         stats = pd.DataFrame({
#                 'count': grouped.count(),
#                 statistic: grouped.agg(statistic)
#         })
#         result.append((q, stats))
#     return result


# def bin_stats(signal, ret, bins=None, quantile=None, method='mean', m=None):

#     return bin_mean_stats(signal, ret, bins, method, m) + \
#         quantile_mean_stats(signal, ret, quantile, method, m)


import numpy as np
import pandas as pd
import plotly.graph_objects as go


def quantile_mean_stats(signal, ret, quantile=None, method='mean'):
    df = pd.qcut(signal, q=quantile)

    # Group and compute statistics
    grouped = ret.groupby(df)
    return pd.DataFrame({
            'count': grouped.count(),
            method: grouped.agg(method)
    })


@dataclass
class QM:
    """ quantile mean """
    q: int
    method: str ='mean'  # or median

    def gen_stats(self, signal: pd.Series, ret: pd.Series) -> pd.DataFrame:
        """ generate stats from data """
        return quantile_mean_stats(signal, ret, self.q, self.method)
    

    def plot(self, signal: pd.Series, ret: pd.Series) -> go.Figure:
        """ plot stats """
        stats = self.gen_stats(signal, ret)
        # Convert interval index to string for plotly compatibility
        stats.index = stats.index.astype(str)
        
        fig = go.Figure(data=[go.Bar(x=stats.index, y=stats[self.method])])
        fig.update_layout(
            title="Signal Quantile Analysis",
            xaxis_title="Signal Quantiles", 
            yaxis_title=f"Return {self.method.title()}"
        )
        return fig
    
    def plot_all_returns_in_one_fig(
            self, 
            signal: pd.Series, 
            ret: pd.DataFrame,  # columns are returns
            ) -> go.Figure:
        """ plot all returns in one figure as different colors """
        stats = self.gen_stats(signal, ret)
        # Convert interval index to string for plotly compatibility
        stats.index = stats.index.astype(str)
        
        fig = go.Figure(data=[
            go.Bar(x=stats.index, y=stats[col], name=col)
            for col in ret.columns
        ])
        fig.update_layout(
            title="Signal Quantile Analysis",
            xaxis_title="Signal Quantiles", 
            yaxis_title=f"Return {self.method.title()}"
        )
        return fig
    
    def plot_all_signals_in_one_fig(
            self,
            signal_df: pd.DataFrame,  # columns are signals
            ret: pd.Series,
            ) -> go.Figure:
        """ plot quantile analysis for multiple signals against one return series
        
        Args:
            signal_df: DataFrame where each column is a different signal
            ret: Series of returns to evaluate signals against
        Returns:
            plotly Figure with bars for each signal
        """
        fig = go.Figure(data=[
            go.Bar(
                x=self.gen_stats(signal_df[col], ret).index.astype(str),
                y=self.gen_stats(signal_df[col], ret)[self.method],
                name=col
            )
            for col in signal_df.columns
        ])
        fig.update_layout(
            title="Multiple Signals Quantile Analysis",
            xaxis_title="Signal Quantiles",
            yaxis_title=f"Return {self.method.title()}",
            barmode='group'  # Groups bars side by side
        )
        return fig
    
def _get_bins(bins: Union[int, np.ndarray], signal: pd.Series, m: float = None):
    if isinstance(bins, np.ndarray):
        return bins
    elif isinstance(bins, int):
        if m is None:
            m = signal.abs().max()
        return np.linspace(-m, m, bins + 1)


@dataclass
class BM:
    bins: Union[int, np.ndarray]
    method: str = 'mean'
    bin_clip: float = None
    confidence_level: float = 0.95
    error_bars: bool = True
    
    def gen_stats(self, signal: pd.Series, ret: pd.Series) -> pd.DataFrame:
        bins = _get_bins(self.bins, signal, self.bin_clip)
        return _get_bin_mean(
            signal, ret, bins, 
            statistic=self.method,
            error_bars=self.error_bars,
            confidence_level=self.confidence_level
        )
    
    def plot(self, signal: pd.Series, ret: pd.Series) -> go.Figure:
        """Generate a side-by-side plot of counts and means/medians for binned analysis"""
        stats_df = self.gen_stats(signal, ret)
        
        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=("Sample Count", "Return Statistics"))
        
        # Convert interval index to string for plotting
        x_labels = [str(interval) for interval in stats_df.index]
        
        # Left subplot - Count
        fig.add_trace(
            go.Bar(x=x_labels, y=stats_df['count'], name='Count'),
            row=1, col=1
        )
        
        # Right subplot - Mean/Median Returns
        if self.method == 'mean' and self.error_bars and 'error_margin' in stats_df:
            fig.add_trace(
                go.Bar(
                    x=x_labels, 
                    y=stats_df[self.method],
                    error_y=dict(
                        type='data',
                        array=stats_df['error_margin'],
                        visible=True
                    ),
                    name=self.method.capitalize()
                ),
                row=1, col=2
            )
        else:
            fig.add_trace(
                go.Bar(x=x_labels, y=stats_df[self.method], 
                      name=self.method.capitalize()),
                row=1, col=2
            )
        
        fig.update_layout(
            title="Binned Analysis",
            showlegend=True
        )
        return fig

    def plot_overlay(self, signal: pd.Series, ret: pd.Series) -> go.Figure:
        """Generate a plot with count bars and mean/median line overlay using secondary axis"""
        stats_df = self.gen_stats(signal, ret)
        
        # Convert interval index to string for plotting
        x_labels = [str(interval) for interval in stats_df.index]
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add count bars
        fig.add_trace(
            go.Bar(x=x_labels, y=stats_df['count'], name='Sample Count'),
            secondary_y=False
        )
        
        # Add mean/median line
        if self.method == 'mean' and self.error_bars and 'error_margin' in stats_df:
            fig.add_trace(
                go.Scatter(
                    x=x_labels, 
                    y=stats_df[self.method],
                    error_y=dict(
                        type='data',
                        array=stats_df['error_margin'],
                        visible=True,
                        color='red'
                    ),
                    name=f'Return {self.method.capitalize()}',
                    line=dict(color='red')
                ),
                secondary_y=True
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=x_labels, 
                    y=stats_df[self.method],
                    name=f'Return {self.method.capitalize()}',
                    line=dict(color='red')
                ),
                secondary_y=True
            )
        
        # Update layout
        fig.update_layout(
            title="Binned Analysis",
            showlegend=True,
            xaxis_title="Bins"
        )
        
        # Update y-axes labels
        fig.update_yaxes(title_text="Sample Count", secondary_y=False)
        fig.update_yaxes(title_text=f"Return {self.method.capitalize()}", secondary_y=True)
        
        return fig
    
    def plot_one_signal_all_returns(self, 
                                  signal: pd.Series, 
                                  returns: pd.DataFrame) -> List[go.Figure]:
        """Generate binned analysis plots for one signal against all return series
        
        Args:
            signal: Single signal series to analyze
            returns: DataFrame containing multiple return series
            
        Returns:
            List of plotly Figures, one for each return series
        """
        figures = []
        
        for col in returns.columns:
            stats_df = self.gen_stats(signal, returns[col])
            x_labels = [str(interval) for interval in stats_df.index]
            
            # Create figure with secondary y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add count bars
            fig.add_trace(
                go.Bar(x=x_labels, y=stats_df['count'], name='Sample Count'),
                secondary_y=False
            )
            
            # Add mean/median line
            if self.method == 'mean' and self.error_bars and 'error_margin' in stats_df:
                fig.add_trace(
                    go.Scatter(
                        x=x_labels, 
                        y=stats_df[self.method],
                        error_y=dict(
                            type='data',
                            array=stats_df['error_margin'],
                            visible=True,
                            color='red'
                        ),
                        name=f'Return {self.method.capitalize()}',
                        line=dict(color='red')
                    ),
                    secondary_y=True
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=x_labels, 
                        y=stats_df[self.method],
                        name=f'Return {self.method.capitalize()}',
                        line=dict(color='red')
                    ),
                    secondary_y=True
                )
            
            fig.update_layout(
                title=f"Binned Analysis: {signal.name} vs {col}",
                showlegend=True,
                xaxis_title="Bins"
            )
            
            fig.update_yaxes(title_text="Sample Count", secondary_y=False)
            fig.update_yaxes(title_text=f"Return {self.method.capitalize()}", secondary_y=True)
            
            figures.append(fig)
            
        return figures

    def plot_all_signals_one_return(self, 
                                  signals: pd.DataFrame, 
                                  ret: pd.Series) -> List[go.Figure]:
        """Generate binned analysis plots for all signals against one return series
        
        Args:
            signals: DataFrame containing multiple signal series
            ret: Single return series to analyze
            
        Returns:
            List of plotly Figures, one for each signal
        """
        figures = []
        
        for col in signals.columns:
            stats_df = self.gen_stats(signals[col], ret)
            x_labels = [str(interval) for interval in stats_df.index]
            
            # Create figure with secondary y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add count bars
            fig.add_trace(
                go.Bar(x=x_labels, y=stats_df['count'], name='Sample Count'),
                secondary_y=False
            )
            
            # Add mean/median line
            if self.method == 'mean' and self.error_bars and 'error_margin' in stats_df:
                fig.add_trace(
                    go.Scatter(
                        x=x_labels, 
                        y=stats_df[self.method],
                        error_y=dict(
                            type='data',
                            array=stats_df['error_margin'],
                            visible=True,
                            color='red'
                        ),
                        name=f'Return {self.method.capitalize()}',
                        line=dict(color='red')
                    ),
                    secondary_y=True
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=x_labels, 
                        y=stats_df[self.method],
                        name=f'Return {self.method.capitalize()}',
                        line=dict(color='red')
                    ),
                    secondary_y=True
                )
            
            fig.update_layout(
                title=f"Binned Analysis: {col} vs {ret.name}",
                showlegend=True,
                xaxis_title="Bins"
            )
            
            fig.update_yaxes(title_text="Sample Count", secondary_y=False)
            fig.update_yaxes(title_text=f"Return {self.method.capitalize()}", secondary_y=True)
            
            figures.append(fig)
            
        return figures
    
