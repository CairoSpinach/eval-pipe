from dataclasses import dataclass
from typing import Union
import pandas as pd
import numpy as np
import plotly.graph_objects as go

@dataclass
class RollingPair:
    lookback: Union[str, pd.Timedelta]
    min_periods: int = '10B' # 10 business days
    
    def beta(self, signal: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Calculate rolling beta between signal and returns.
        
        Args:
            signal: Input signal series with datetime index
            returns: Return series with datetime index
            
        Returns:
            Daily resampled rolling beta series
        """
        # Ensure indexes are aligned
        df = pd.DataFrame({'signal': signal, 'returns': returns})
        df = df.dropna()
        
        # Calculate rolling covariance and variance
        roll_cov = df.rolling(window=self.lookback, min_periods=self.min_periods).cov()
        roll_var = df['signal'].rolling(window=self.lookback, min_periods=self.min_periods).var()
        
        # Extract beta values (cov/var) and resample to business days
        beta = (roll_cov.xs('returns', level=1)['signal'] / roll_var)
        return beta.resample('B').last()
    
    def corr(self, signal: pd.Series, returns: pd.Series, method='pearson') -> pd.Series:
        """
        Calculate rolling correlation between signal and returns.
        
        Args:
            signal: Input signal series with datetime index
            returns: Return series with datetime index
            
        Returns:
            Daily resampled rolling correlation series
        """
        # Ensure indexes are aligned
        df = pd.DataFrame({'signal': signal, 'returns': returns})
        df = df.dropna()
        
        # Calculate rolling correlation and resample to business days
        roll_corr = df['signal'].rolling(window=self.lookback, min_periods=self.min_periods).corr(df['returns'], method=method)
        return roll_corr.resample('B').last()
    
    def plot(self, signal: pd.Series, returns: pd.Series, plot_type: str = 'beta') -> go.Figure:
        """Plot rolling analysis between signal and returns.
        
        Args:
            signal: Input signal series
            returns: Return series
            plot_type: Type of analysis to plot ('beta', 'corr', or 'rank-corr')
        """
        if plot_type == 'beta':
            series = self.beta(signal, returns)
            title = "Rolling Beta Analysis"
            y_label = "Beta"
        elif plot_type == 'corr':
            series = self.corr(signal, returns, method='pearson')
            title = "Rolling Correlation Analysis"
            y_label = "Correlation"
        elif plot_type == 'rank-corr':
            series = self.corr(signal, returns, method='spearman')
            title = "Rolling Rank Correlation Analysis"
            y_label = "Rank Correlation"
        else:
            raise ValueError("plot_type must be one of: 'beta', 'corr', 'rank-corr'")
        
        fig = go.Figure(data=[go.Scatter(x=series.index, y=series.values)])
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title=y_label
        )
        return fig
    
    def plot_all_returns_in_one_fig(self, signal: pd.Series, returns: pd.DataFrame, plot_type: str = 'beta') -> go.Figure:
        """Plot rolling analysis for one signal against multiple return series.
        
        Args:
            signal: Input signal series
            returns: DataFrame of return series
            plot_type: Type of analysis to plot ('beta', 'corr', or 'rank-corr')
        """
        if plot_type == 'beta':
            method = self.beta
            title = "Rolling Beta Analysis - Multiple Returns"
            y_label = "Beta"
        elif plot_type == 'corr':
            method = lambda s, r: self.corr(s, r, method='pearson')
            title = "Rolling Correlation Analysis - Multiple Returns"
            y_label = "Correlation"
        elif plot_type == 'rank-corr':
            method = lambda s, r: self.corr(s, r, method='spearman')
            title = "Rolling Rank Correlation Analysis - Multiple Returns"
            y_label = "Rank Correlation"
        else:
            raise ValueError("plot_type must be one of: 'beta', 'corr', 'rank-corr'")
        
        fig = go.Figure(data=[
            go.Scatter(
                x=method(signal, returns[col]).index,
                y=method(signal, returns[col]).values,
                name=col
            )
            for col in returns.columns
        ])
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title=y_label
        )
        return fig
    
    def plot_all_signals_in_one_fig(self, signals: pd.DataFrame, returns: pd.Series, plot_type: str = 'beta') -> go.Figure:
        """Plot rolling analysis for multiple signals against one return series.
        
        Args:
            signals: DataFrame of signal series
            returns: Return series
            plot_type: Type of analysis to plot ('beta', 'corr', or 'rank-corr')
        """
        if plot_type == 'beta':
            method = self.beta
            title = "Rolling Beta Analysis - Multiple Signals"
            y_label = "Beta"
        elif plot_type == 'corr':
            method = lambda s, r: self.corr(s, r, method='pearson')
            title = "Rolling Correlation Analysis - Multiple Signals"
            y_label = "Correlation"
        elif plot_type == 'rank-corr':
            method = lambda s, r: self.corr(s, r, method='spearman')
            title = "Rolling Rank Correlation Analysis - Multiple Signals"
            y_label = "Rank Correlation"
        else:
            raise ValueError("plot_type must be one of: 'beta', 'corr', 'rank-corr'")
        
        fig = go.Figure(data=[
            go.Scatter(
                x=method(signals[col], returns).index,
                y=method(signals[col], returns).values,
                name=col
            )
            for col in signals.columns
        ])
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title=y_label
        )
        return fig

@dataclass
class PeriodicPair:
    period: str  # e.g., '2W', '3M'
    
    def beta(self, signal: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Calculate periodic beta between signal and returns.
        
        Args:
            signal: Input signal series with datetime index
            returns: Return series with datetime index
            
        Returns:
            Beta series with values at the end of each period
        """
        # Ensure indexes are aligned
        df = pd.DataFrame({'signal': signal, 'returns': returns})
        df = df.dropna()
        
        # Resample to periods and calculate beta for each period
        grouped = df.groupby(pd.Grouper(freq=self.period))
        
        betas = []
        for _, period_df in grouped:
            if len(period_df) >= 2:  # Need at least 2 points
                cov = period_df['signal'].cov(period_df['returns'])
                var = period_df['signal'].var()
                beta = cov / var if var != 0 else np.nan
                betas.append((period_df.index[-1], beta))
                
        return pd.Series({t: b for t, b in betas})
    
    def corr(self, signal: pd.Series, returns: pd.Series, method='pearson') -> pd.Series:
        """
        Calculate periodic correlation between signal and returns.
        
        Args:
            signal: Input signal series with datetime index
            returns: Return series with datetime index
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Correlation series with values at the end of each period
        """
        # Ensure indexes are aligned
        df = pd.DataFrame({'signal': signal, 'returns': returns})
        df = df.dropna()
        
        # Resample to periods and calculate correlation for each period
        grouped = df.groupby(pd.Grouper(freq=self.period))
        
        corrs = []
        for _, period_df in grouped:
            if len(period_df) >= 2:  # Need at least 2 points
                corr = period_df['signal'].corr(period_df['returns'], method=method)
                corrs.append((period_df.index[-1], corr))
                
        return pd.Series({t: c for t, c in corrs})
    
    def plot(self, signal: pd.Series, returns: pd.Series, plot_type: str = 'beta') -> go.Figure:
        """Plot periodic analysis between signal and returns.
        
        Args:
            signal: Input signal series
            returns: Return series
            plot_type: Type of analysis to plot ('beta', 'corr', or 'rank-corr')
        """
        if plot_type == 'beta':
            series = self.beta(signal, returns)
            title = f"Periodic Beta Analysis ({self.period})"
            y_label = "Beta"
        elif plot_type == 'corr':
            series = self.corr(signal, returns, method='pearson')
            title = f"Periodic Correlation Analysis ({self.period})"
            y_label = "Correlation"
        elif plot_type == 'rank-corr':
            series = self.corr(signal, returns, method='spearman')
            title = f"Periodic Rank Correlation Analysis ({self.period})"
            y_label = "Rank Correlation"
        else:
            raise ValueError("plot_type must be one of: 'beta', 'corr', 'rank-corr'")
        
        fig = go.Figure(data=[go.Scatter(x=series.index, y=series.values, mode='lines+markers')])
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title=y_label
        )
        return fig
    
    def plot_all_returns_in_one_fig(self, signal: pd.Series, returns: pd.DataFrame, plot_type: str = 'beta') -> go.Figure:
        """Plot periodic analysis for one signal against multiple return series.
        
        Args:
            signal: Input signal series
            returns: DataFrame of return series
            plot_type: Type of analysis to plot ('beta', 'corr', or 'rank-corr')
        """
        if plot_type == 'beta':
            method = self.beta
            title = f"Periodic Beta Analysis - Multiple Returns ({self.period})"
            y_label = "Beta"
        elif plot_type == 'corr':
            method = lambda s, r: self.corr(s, r, method='pearson')
            title = f"Periodic Correlation Analysis - Multiple Returns ({self.period})"
            y_label = "Correlation"
        elif plot_type == 'rank-corr':
            method = lambda s, r: self.corr(s, r, method='spearman')
            title = f"Periodic Rank Correlation Analysis - Multiple Returns ({self.period})"
            y_label = "Rank Correlation"
        else:
            raise ValueError("plot_type must be one of: 'beta', 'corr', 'rank-corr'")
        
        fig = go.Figure(data=[
            go.Scatter(
                x=method(signal, returns[col]).index,
                y=method(signal, returns[col]).values,
                name=col,
                mode='lines+markers'
            )
            for col in returns.columns
        ])
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title=y_label
        )
        return fig
    
    def plot_all_signals_in_one_fig(self, signals: pd.DataFrame, returns: pd.Series, plot_type: str = 'beta') -> go.Figure:
        """Plot periodic analysis for multiple signals against one return series.
        
        Args:
            signals: DataFrame of signal series
            returns: Return series
            plot_type: Type of analysis to plot ('beta', 'corr', or 'rank-corr')
        """
        if plot_type == 'beta':
            method = self.beta
            title = f"Periodic Beta Analysis - Multiple Signals ({self.period})"
            y_label = "Beta"
        elif plot_type == 'corr':
            method = lambda s, r: self.corr(s, r, method='pearson')
            title = f"Periodic Correlation Analysis - Multiple Signals ({self.period})"
            y_label = "Correlation"
        elif plot_type == 'rank-corr':
            method = lambda s, r: self.corr(s, r, method='spearman')
            title = f"Periodic Rank Correlation Analysis - Multiple Signals ({self.period})"
            y_label = "Rank Correlation"
        else:
            raise ValueError("plot_type must be one of: 'beta', 'corr', 'rank-corr'")
        
        fig = go.Figure(data=[
            go.Scatter(
                x=method(signals[col], returns).index,
                y=method(signals[col], returns).values,
                name=col,
                mode='lines+markers'
            )
            for col in signals.columns
        ])
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title=y_label
        )
        return fig
