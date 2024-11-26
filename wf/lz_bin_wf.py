from dataclasses import dataclass
from typing import List, Optional, Union
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from wf.utils import create_grid_layout


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
    
    def plot_side_by_side(self, signal: pd.Series, ret: pd.Series) -> go.Figure:
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
    

class BMReport:

    @staticmethod
    def gen_report_single(signal: pd.Series, ret: pd.Series, bins_list: List[int],
                   method: str = 'mean', bin_clip: float = None,
                   confidence_level: float = 0.95, error_bars: bool = True,
                   plot_type: str = 'side-by-side', plot_col_wrap: int = 2) -> str:
        # Handle single bin number or list of bin numbers
        figures = []
        for bins in bins_list:
            # Assert bins is a positive integer
            assert isinstance(bins, int) and bins > 0, f"bins must be a positive integer, got {bins}"
            
            # Create BM analyzer for this bin size
            bm = BM(
                bins=bins,
                method=method,
                bin_clip=bin_clip,
                confidence_level=confidence_level,
                error_bars=error_bars
            )
            
            # Generate figure based on plot type
            
            fig = bm.plot_overlay(signal, ret) if plot_type == 'overlay' else bm.plot_side_by_side(signal, ret)
            fig.update_layout(title=f"{bins} Bins Analysis")
            figures.append(fig)
            
            # Create subsection for this bin size
        subsection = f"""
            <div style="margin-bottom: 20px;">
                <h4>{bins} Bins Analysis</h4>
                {create_grid_layout(figures, cols=plot_col_wrap)}
            </div>
        """
        
        # Combine all subsections into the BM section
        return subsection
    
    @staticmethod
    def create_multi_returns_section(
        signal: pd.Series,
        ret_df: pd.DataFrame,
        bins: Union[int, List[int]] = 20,
        method: str = "mean",
        bin_clip: Optional[float] = None,
        confidence_level: float = 0.95,
        error_bars: bool = True,
        plot_type: str = "side-by-side",
        plot_col_wrap: int = 2
    ) -> str:
        """Create the binned means analysis section for single signal and multiple returns"""
        subsections = []
        
        # Handle single bin number or list of bin numbers
        bins_list = [bins] if isinstance(bins, (int, np.ndarray)) else bins
        
        for bin_count in bins_list:
            # Create BM analyzer for this bin size
            bm = BM(
                bins=bin_count,
                method=method,
                bin_clip=bin_clip,
                confidence_level=confidence_level,
                error_bars=error_bars
            )
            
            # Get figures for this bin size
            figures = bm.plot_one_signal_all_returns(
                signal=signal,
                returns=ret_df,
            )
            
            subsection = f"""
                <div style="margin-bottom: 20px;">
                    {create_grid_layout(figures, cols=plot_col_wrap, title=f"{bin_count} Bins", title_level=4)}
                </div>
            """
            subsections.append(subsection)
        
        return f"""
            <div style="margin-bottom: 30px;">
                <h3>Bin Mean Analysis</h3>
                {"".join(subsections)}
            </div>
        """

    @staticmethod
    def create_multi_signals_section(
        signal_df: pd.DataFrame,
        ret: pd.Series,
        bins: Union[int, List[int]] = 20,
        method: str = "mean",
        bin_clip: Optional[float] = None,
        confidence_level: float = 0.95,
        error_bars: bool = True,
        plot_col_wrap: int = 2
    ) -> str:
        """Create the binned means analysis section for multiple signals and single return
        each bins values would be itself a subsection. Multiple signals should share a subsection
        (in multi-figure)
        
        """
        subsections = []
        
        # Handle single bin number or list of bin numbers
        bins_list = [bins] if isinstance(bins, (int, np.ndarray)) else bins
        
        for bin_count in bins_list:
            # Create BM analyzer for this bin size
            bm = BM(
                bins=bin_count,
                method=method,
                bin_clip=bin_clip,
                confidence_level=confidence_level,
                error_bars=error_bars
            )
            
            # Get figures for this bin size
            figures = bm.plot_all_signals_one_return(
                signal_df=signal_df,
                ret=ret,
            )
            
            subsection = f"""
                <div style="margin-bottom: 20px;">
                    {create_grid_layout(figures, cols=plot_col_wrap, title=f"{bin_count} Bins", title_level=4)}
                </div>
            """
            subsections.append(subsection)
        
        return f"""
            <div style="margin-bottom: 30px;">
                <h3>Bin Mean Analysis</h3>
                {"".join(subsections)}
            </div>
        """ 