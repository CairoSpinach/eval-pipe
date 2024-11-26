import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Union, Literal

class PairStats:
    def __init__(self, stats_df: pd.DataFrame):
        """Initialize with a DataFrame containing bin statistics"""
        self.stats_df = stats_df
        
    def plot(self, title: str = "Binned Statistics") -> go.Figure:
        """Generate a side-by-side plot of counts and means/medians"""
        fig = make_subplots(rows=1, cols=2, 
                           subplot_titles=("Sample Count", "Return Statistics"))
        
        # Left subplot - Count
        
        # Convert interval index to string for plotting
        x_labels = [str(interval) for interval in self.stats_df.index]
        
        fig.add_trace(
            go.Bar(x=x_labels, y=self.stats_df['count'], name='Count'),
            row=1, col=1
        )
        
        # Right subplot - Mean/Median Returns
        value_col = next(col for col in ['mean', 'median'] if col in self.stats_df.columns)
        fig.add_trace(
            go.Bar(x=x_labels, y=self.stats_df[value_col], 
                  name=value_col.capitalize()),
            row=1, col=2
        )
        
        fig.update_layout(title=title, showlegend=True)
        return fig
    
    def gen_report(self, bin_fig: go.Figure = None, quantile_fig: go.Figure = None) -> str:
        """Generate an HTML report with table and plots
        
        Parameters:
            bin_fig: Optional figure from bin_mean analysis
            quantile_fig: Optional figure from quantile_mean analysis
        """
        table_html = self.stats_df.to_html(classes='table table-striped', 
                                         float_format=lambda x: f'{x:.4f}')
        
        print('gen report')
        # Default to standard plot if no figures provided
        if bin_fig is None and quantile_fig is None:
            fig_html = self.plot().to_html(full_html=False)
            plot_div = f'<div class="mt-4">{fig_html}</div>'
            print('regenerated plot')
        else:
            # Create side-by-side plot divs if both figures provided
            plots = []
            if bin_fig is not None:
                plots.append(bin_fig.to_html(full_html=False))
            if quantile_fig is not None:
                plots.append(quantile_fig.to_html(full_html=False))
                
            # Create a 2-column layout for the plots using Bootstrap grid classes
            # plots[0] goes in left column, plots[1] in right column if both exist
            plot_div = """
            <div class="row mt-4">
                <div class="col-md-6">
                    {0}
                </div>
                <div class="col-md-6">
                    {1}
                </div>
            </div>
            """.format(*plots)
            
        return f"""
        <html>
            <head>
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            </head>
            <body class="container mt-5">
                <h2>Statistics Report</h2>
                {table_html}
                {plot_div}
            </body>
        </html>
        """
    

    def __repr__(self):
        """Forward repr to stats_df"""
        return repr(self.stats_df)
    
    def __str__(self):
        """Forward str to stats_df"""
        return str(self.stats_df)


def bin_mean(signal: pd.Series, 
            returns: pd.Series,
            bins: Union[int, list] = 10,
            statistic: Literal['mean', 'median'] = 'mean') -> PairStats:
    """
    Generate binned statistics for signal-return pairs
    
    Parameters:
        signal: Input signal to bin
        returns: Corresponding returns
        bins: Number of bins or list of bin edges
        statistic: Whether to use 'mean' or 'median' for return statistics
    """
    # Create bins and labels
    if isinstance(bins, int):
        bins = np.linspace(signal.min(), signal.max(), bins + 1)
    
    # Compute bin assignments
    # labels = bins[:-1]
    
    binned = pd.cut(signal, bins=bins)
    
    # Group and compute statistics
    grouped = returns.groupby(binned)
    stats = {
        'count': grouped.count(),
        statistic: grouped.agg(statistic)
    }
    
    return PairStats(pd.DataFrame(stats))

def quantile_mean(signal: pd.Series,
                 returns: pd.Series,
                 q: int = 10,
                 statistic: Literal['mean', 'median'] = 'mean') -> PairStats:
    """
    Generate quantile-based statistics for signal-return pairs
    
    Parameters:
        signal: Input signal to bin
        returns: Corresponding returns
        q: Number of quantiles
        statistic: Whether to use 'mean' or 'median' for return statistics
    """
    # Compute quantile assignments
    quantiles = pd.qcut(signal, q=q)
    
    # Group and compute statistics
    grouped = returns.groupby(quantiles)
    stats = {
        'count': grouped.count(),
        statistic: grouped.agg(statistic)
    }
    
    return PairStats(pd.DataFrame(stats))
