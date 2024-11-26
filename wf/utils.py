from datetime import datetime, time, date
import pandas as pd
import pandas_market_calendars as mcal
from typing import List, Union
import plotly.graph_objects as go

def create_business_hours_index(
    start_date: Union[str, datetime, date],
    end_date: Union[str, datetime, date],
    start_time: Union[str, time] = "09:30",
    end_time: Union[str, time] = "16:00",
    freq: Union[str, pd.Timedelta] = "1min",
    calendar: str = "SIFMA_US"
) -> pd.DatetimeIndex:
    """
    Create a DatetimeIndex for business hours between start and end dates.
    
    Parameters:
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
        start_time: Daily start time (default market open "09:30")
        end_time: Daily end time (default market close "16:00")
        freq: Frequency of the index (default "1min")
        calendar: Market calendar to use (default "NYSE")
    """
    # Convert frequency to pd.Timedelta if string
    if isinstance(freq, str):
        freq = pd.Timedelta(freq)
    
    # Convert inputs to pd.Timestamp
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    
    if isinstance(start_time, str):
        start_time = pd.Timestamp(start_time).time()
    if isinstance(end_time, str):
        end_time = pd.Timestamp(end_time).time()
    
    # Get market calendar
    market_cal = mcal.get_calendar(calendar)
    
    # Get schedule
    schedule = market_cal.schedule(
        start_date=start_ts,
        end_date=end_ts
    )
    
    all_datetimes = []
    
    # Process each trading day
    for date in schedule.index:
        # Use provided times unless overridden by early close
        close_time = min(end_time, schedule.loc[date, 'market_close'].time())
        
        daily_range = pd.date_range(
            start=pd.Timestamp.combine(date.date(), start_time),
            end=pd.Timestamp.combine(date.date(), close_time),
            freq=freq,
            inclusive='left'
        )
        all_datetimes.extend(daily_range)
    
    return pd.DatetimeIndex(all_datetimes)

def create_bond_market_index(
    start_date: str,
    end_date: str,
    freq: str = "1min",
    calendar_name: str = "SIFMA"
) -> pd.DatetimeIndex:
    """
    Create DatetimeIndex for US bond market hours using SIFMA calendar.
    Handles early closes and holidays.
    
    Parameters:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        freq: Frequency of the index (default "1min")
        calendar_name: Name of the calendar to use (default "SIFMA")
    """
    # Get SIFMA calendar
    calendar = mcal.get_calendar(calendar_name) if calendar_name else mcal.get_calendar('SIFMA')
    
    # Get schedule including early closes
    schedule = calendar.schedule(
        start_date=start_date,
        end_date=end_date
    )
    
    all_datetimes = []
    
    # Process each trading day
    for _, row in schedule.iterrows():
        daily_range = pd.date_range(
            start=row['market_open'],
            end=row['market_close'],
            freq=freq,
            inclusive='left'  # Don't include market close time
        )
        all_datetimes.extend(daily_range)
    
    return pd.DatetimeIndex(all_datetimes)


def create_grid_layout_deprecate(figures: List[go.Figure], 
                      cols: int = 2, 
                      title: str = "Analysis Grid") -> str:
    """Create a responsive grid layout HTML string for multiple plotly figures
    
    Args:
        figures: List of plotly Figure objects
        cols: Number of columns in the grid
        title: Title for the grid layout
        
    Returns:
        HTML string with Bootstrap grid layout containing the figures
    """
    # Convert figures to HTML
    plot_htmls = [fig.to_html(full_html=False, include_plotlyjs=False) for fig in figures]
    
    # Calculate bootstrap column width
    col_width = 12 // cols  # Bootstrap uses 12-column system
    
    # Create row divs with plot divs inside
    rows = []
    for i in range(0, len(plot_htmls), cols):
        # Get plots for this row (up to cols number)
        row_plots = plot_htmls[i:i + cols]
        
        # Create columns for each plot
        col_divs = [
            f'<div class="col-md-{col_width}">{plot}</div>'
            for plot in row_plots
        ]
        
        # Add empty columns if row is not full
        remaining = cols - len(row_plots)
        if remaining > 0:
            col_divs.extend([f'<div class="col-md-{col_width}"></div>'] * remaining)
            
        # Join columns into a row
        row = f'<div class="row mb-4">{"".join(col_divs)}</div>'
        rows.append(row)
    
    # Combine all rows into final layout
    return f"""
    <div class="container-fluid mt-4">
        <h2 class="mb-4">{title}</h2>
        {"".join(rows)}
    </div>
    """


def create_grid_layout(figures: List[go.Figure], 
                      cols: int = 2, 
                      title: str = None, title_level=3) -> str:
    """Create a responsive grid layout HTML string for multiple plotly figures
    
    Args:
        figures: List of plotly Figure objects
        cols: Number of columns in the grid
        title: Title for the grid layout
        
    Returns:
        HTML string with CSS Grid layout containing the figures
    """
    # Convert figures to HTML
    plot_htmls = [fig.to_html(full_html=False, include_plotlyjs=False) for fig in figures]
    
    # Create plot divs
    plot_divs = [f'<div class="plot-container">{plot}</div>' for plot in plot_htmls]
    
    grid_title = f'<h{title_level}>{title}</h{title_level}>' if title else ''

    # Combine all plots into grid container
    return f"""
    <div class="grid-container">
        {grid_title}
        <div class="plot-grid">
            {"".join(plot_divs)}
        </div>
    </div>
    <style>
        .grid-container {{
            padding: 20px;
            max-width: 100%;
        }}
        .plot-grid {{
            display: grid;
            grid-template-columns: repeat({cols}, 1fr);
            gap: 20px;
            width: 100%;
        }}
        .plot-container {{
            min-width: 0;  /* Prevents plots from overflowing */
        }}
        @media (max-width: 1200px) {{
            .plot-grid {{
                grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            }}
        }}
    </style>
    """


def write_html(html_content: str, output_file: str = "output.html") -> None:
    """Write HTML content to a file
    
    Args:
        html_content: HTML string to write
        output_file: Path to output HTML file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)


if __name__ == "__main__":
    # Generate index for a week
    idx = create_bond_market_index(
        "2024-01-02",  # Tuesday after New Year
        "2024-01-05"
    )
    
    # Print sample information
    print("Bond Market Hours Sample:")
    print(f"First timestamp: {idx[0]}")
    print(f"Last timestamp: {idx[-1]}")
    print(f"Total points: {len(idx)}")
    
    # Show schedule for the period
    calendar = mcal.get_calendar('SIFMA')
    schedule = calendar.schedule(
        start_date="2024-01-02",
        end_date="2024-01-05"
    )
    print("\nTrading Schedule:")
    print(schedule)
    
    # Demonstrate early close handling
    # Let's find a date with early close (e.g., day before Independence Day)
    july_3_idx = create_bond_market_index(
        "2024-07-03",
        "2024-07-03"
    )
    
    print("\nEarly Close Example (July 3, 2024):")
    print(f"First timestamp: {july_3_idx[0]}")
    print(f"Last timestamp: {july_3_idx[-1]}")
    print(f"Total points: {len(july_3_idx)}")
