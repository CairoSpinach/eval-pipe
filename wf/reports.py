from typing import List, Union, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
from IPython.display import HTML
from .utils import create_grid_layout
from .lz_bin_wf import QM, BM

@dataclass
class ReportConfig:
    
    reports: List[str] = None   # reports options: "bm", "qm" 
    method: str = "mean"  # method options: "mean", "median"
    bins: Optional[Union[int, List[float]]] = 20
    bin_clip: Optional[float] = None
    quantiles: List[int] = None
    plot_type: str = "side-by-side"  # plot_type options: "side-by-side", "overlay"
    confidence_level: float = 0.95
    error_bars: bool = True


def signal_report(
    signal: pd.Series,
    ret: pd.Series,
    config: ReportConfig = None
) -> str:
    """Generate an HTML report analyzing signal characteristics using BM and QM analysis
    
    Args:
        signal: Input signal series
        ret: Return series to analyze against
        config: ReportConfig object with analysis parameters
        
    Returns:
        HTML string containing the report
    """
    if config is None:
        config = ReportConfig(
            reports=["bm", "qm"],
            method="mean",
            bins=20,
            bin_clip=None,
            quantiles=[5, 10],
            confidence_level=0.95,
            error_bars=True
        )
    
    sections = []
    
    # Stats section
    stats_section = f"""
        <div style="margin-bottom: 20px;">
            <h3>Signal Statistics:</h3>
            <table style="width: auto; border-collapse: collapse; margin: 10px 0;">
                <tr>
                    <td style="padding: 5px; border: 1px solid #ddd;"><strong>Mean:</strong></td>
                    <td style="padding: 5px; border: 1px solid #ddd;">{signal.mean():.4f}</td>
                    <td style="padding: 5px; border: 1px solid #ddd;"><strong>Std:</strong></td>
                    <td style="padding: 5px; border: 1px solid #ddd;">{signal.std():.4f}</td>
                </tr>
                <tr>
                    <td style="padding: 5px; border: 1px solid #ddd;"><strong>Min:</strong></td>
                    <td style="padding: 5px; border: 1px solid #ddd;">{signal.min():.4f}</td>
                    <td style="padding: 5px; border: 1px solid #ddd;"><strong>Max:</strong></td>
                    <td style="padding: 5px; border: 1px solid #ddd;">{signal.max():.4f}</td>
                </tr>
            </table>
        </div>
    """
    sections.append(stats_section)
    
    if "bm" in config.reports:
        # Create binned means analysis
        bm_figures = []
        
        # Handle single bin number or list of bin numbers
        bins_list = [config.bins] if isinstance(config.bins, (int, np.ndarray)) else config.bins
        
        for bins in bins_list:
            # Assert bins is a positive integer
            assert isinstance(bins, int) and bins > 0, f"bins must be a positive integer, got {bins}"
            
            bm = BM(
                bins=bins,
                method=config.method,
                bin_clip=config.bin_clip,
                confidence_level=config.confidence_level,
                error_bars=config.error_bars
            )
            
            # Generate figure based on plot type
            if config.plot_type == 'overlay':
                fig = bm.plot_overlay(signal, ret)
            else:
                fig = bm.plot(signal, ret)
                
            # Update title to include number of bins
            fig.update_layout(title=f"Binned Analysis ({bins} bins)")
            bm_figures.append(fig)
        
        bm_section = f"""
            <div style="margin-bottom: 30px;">
                <h3>Bin Mean Analysis</h3>
                {create_grid_layout(bm_figures, cols=2)}
            </div>
        """
        sections.append(bm_section)
    
    if "qm" in config.reports and config.quantiles:
        # Create quantile analysis for each specified quantile
        qm_figures = []
        for q in config.quantiles:
            qm = QM(q=q, method=config.method)
            fig = qm.plot(signal, ret)
            fig.update_layout(title=f"Quantile Analysis ({q} quantiles)")
            qm_figures.append(fig)
        
        qm_section = f"""
            <div style="margin-bottom: 30px;">
                <h3>Quantile Analysis</h3>
                {create_grid_layout(qm_figures, cols=2)}
            </div>
        """
        sections.append(qm_section)
    
    # Combine all sections
    html_content = f"""
    <div style="padding: 20px;">
        <h1>Signal Analysis Report</h1>
        {"".join(sections)}
    </div>
    """
    
    return html_content


# Test configurations
test_configs = {
    "basic": ReportConfig(
        reports=["bm", "qm"],
        bins=[10, 20],  # Test different bin sizes
        quantiles=[5, 10],  # Test different quantile divisions
        method="mean",
        plot_type="split"  # Default split view
    ),
    
    "overlay_only": ReportConfig(
        reports=["bm"],
        bins=[15],
        quantiles=[],  # No quantile analysis
        method="mean", 
        plot_type="overlay",  # Test overlay view
        error_bars=True,
        confidence_level=0.95
    ),
    
    "quantile_only": ReportConfig(
        reports=["qm"],
        bins=[],  # No bin analysis
        quantiles=[4, 8, 12],  # Test multiple quantile divisions
        method="median"  # Test median instead of mean
    ),
    
    "detailed_bins": ReportConfig(
        reports=["bm"],
        bins=[10, 20, 50],  # Test range of bin sizes
        quantiles=[],
        method="mean",
        plot_type="split",
        bin_clip=3.0,  # Test bin clipping
        error_bars=True,
        confidence_level=0.99  # Test different confidence level
    ),
    
    "comprehensive": ReportConfig(
        reports=["bm", "qm"],
        bins=[15, 30],
        quantiles=[5, 10],
        method="mean",
        plot_type="overlay",
        bin_clip=2.5,
        error_bars=True,
        confidence_level=0.90
    )
}
