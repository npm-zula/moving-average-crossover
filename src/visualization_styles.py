"""
Visualization styles module for consistent styling across all charts

This module provides color palettes, themes, and styling functions to ensure
consistent and professional-looking visualizations throughout the application.
"""

# Modern color palette with carefully selected colors for better visualization
# These colors have been chosen for their contrast, accessibility and aesthetic appeal
COLORS = {
    'background': '#f8f9fa',      # Light background that's easy on the eyes
    'grid': '#e9ecef',            # Subtle grid lines
    'text': '#343a40',            # Dark text for good readability
    'price': '#212529',           # Dark color for price data
    'short_ma': '#4361ee',        # Vibrant blue for short MA
    'long_ma': '#ef476f',         # Pink/red for long MA
    'buy_signal': '#06d6a0',      # Bright green for buy signals
    'sell_signal': '#e63946',     # Bright red for sell signals
    'portfolio': '#118ab2',       # Blue for portfolio value
    'signal_line': '#7209b7',     # Purple for signal line

    # Additional colors for strategy comparison
    'strategy1': '#06d6a0',       # Green
    'strategy2': '#118ab2',       # Blue
    'strategy3': '#7209b7',       # Purple
    'strategy4': '#ef476f',       # Pink/red
    'strategy5': '#ff9e00',       # Orange

    # Neutral colors for backgrounds and accents
    'light_bg': '#f8f9fa',        # Very light background
    'medium_bg': '#e9ecef',       # Medium light background
    'dark_bg': '#ced4da',         # Darker background for contrast
    'highlight': '#ffd166',       # Yellow highlight

    # LinkedIn-specific colors
    'linkedin_blue': '#0077b5',   # LinkedIn brand blue
    'linkedin_accent': '#00a0dc',  # LinkedIn lighter blue
    'linkedin_dark': '#313335',   # LinkedIn dark gray
    'linkedin_light': '#f3f6f8',  # LinkedIn light background
}

# Theme settings for consistent plotting
THEME = {
    'font_family': 'Arial, sans-serif',
    'title_font_size': 20,
    'title_font_color': COLORS['text'],
    'subtitle_font_size': 16,
    'subtitle_font_color': COLORS['text'],
    'axis_title_font_size': 14,
    'axis_title_font_color': COLORS['text'],
    'axis_tick_font_size': 12,
    'axis_tick_font_color': COLORS['text'],
    'legend_font_size': 12,
    'legend_font_color': COLORS['text'],
    'annotation_font_size': 12,
    'annotation_font_color': COLORS['text'],
}

# Plot layout settings
LAYOUT = {
    'plot_bgcolor': COLORS['background'],
    'paper_bgcolor': COLORS['background'],
    'font': {
        'family': THEME['font_family'],
        'color': THEME['title_font_color'],
    },
    'margin': {'l': 50, 'r': 50, 't': 60, 'b': 50},
    'hovermode': 'x unified',
    'legend': {
        'orientation': 'h',
        'yanchor': 'bottom',
        'y': 1.02,
        'xanchor': 'right',
        'x': 1,
        'font': {'size': THEME['legend_font_size'], 'color': THEME['legend_font_color']},
        'bgcolor': COLORS['background'],
        'borderwidth': 0,
    },
    'xaxis': {
        'gridcolor': COLORS['grid'],
        'zerolinecolor': COLORS['grid'],
        'title': {'font': {'size': THEME['axis_title_font_size'], 'color': THEME['axis_title_font_color']}},
        'tickfont': {'size': THEME['axis_tick_font_size'], 'color': THEME['axis_tick_font_color']},
    },
    'yaxis': {
        'gridcolor': COLORS['grid'],
        'zerolinecolor': COLORS['grid'],
        'title': {'font': {'size': THEME['axis_title_font_size'], 'color': THEME['axis_title_font_color']}},
        'tickfont': {'size': THEME['axis_tick_font_size'], 'color': THEME['axis_tick_font_color']},
    },
}

# Strategy color mapping for comparison charts
STRATEGY_COLORS = {
    '5_20': COLORS['strategy1'],
    '10_30': COLORS['strategy2'],
    '20_50': COLORS['strategy3'],
    '50_200': COLORS['strategy4'],
}

# Custom button style for animation controls
BUTTON_STYLE = {
    'bgcolor': COLORS['dark_bg'],
    'bordercolor': COLORS['text'],
    'font': {'size': 14, 'color': COLORS['text']},
    'active': {'bgcolor': COLORS['highlight']}
}

# Custom slider style for animation timeline
SLIDER_STYLE = {
    'bgcolor': COLORS['medium_bg'],
    'bordercolor': COLORS['grid'],
    'borderwidth': 1,
    'tickwidth': 1,
    'tickcolor': COLORS['text'],
    'minorticklen': 5,
    'currentvalue': {
        'font': {'size': 14, 'color': COLORS['text']},
        'prefix': 'Date: ',
        'xanchor': 'right',
        'offset': 10,
    },
}

# Table styles for metrics display
TABLE_STYLE = {
    'header': {
        'fill_color': COLORS['dark_bg'],
        'align': 'center',
        'font': {'size': 14, 'color': COLORS['text'], 'family': THEME['font_family']},
        'height': 40,
    },
    'cells': {
        # Alternating row colors
        'fill_color': [[COLORS['light_bg'], COLORS['medium_bg']] * 50],
        'align': ['left', 'right'],
        'font': {'size': 12, 'color': COLORS['text'], 'family': THEME['font_family']},
        'height': 30,
        'line': {'color': COLORS['grid'], 'width': 1},
    },
}

# Default strategy configurations
DEFAULT_STRATEGIES = [
    {'short_window': 20, 'long_window': 50},
    {'short_window': 50, 'long_window': 200},
    {'short_window': 10, 'long_window': 30},
    {'short_window': 5, 'long_window': 20},
]

# Strategies for animated comparison (includes colors)
ANIMATION_COMPARISON_STRATEGIES = [
    {'short_window': 5, 'long_window': 20, 'color': COLORS['strategy1']},
    {'short_window': 10, 'long_window': 30, 'color': COLORS['strategy2']},
    {'short_window': 20, 'long_window': 50, 'color': COLORS['strategy3']},
    {'short_window': 50, 'long_window': 200, 'color': COLORS['strategy4']},
]

# Strategies for LinkedIn comparison cards (includes names and colors)
LINKEDIN_COMPARISON_STRATEGIES = [
    {'short_window': 5, 'long_window': 20,
        'name': 'Fast (5/20)', 'color': COLORS['strategy1']},
    {'short_window': 20, 'long_window': 50,
        'name': 'Medium (20/50)', 'color': COLORS['strategy3']},
    {'short_window': 50, 'long_window': 200,
        'name': 'Slow (50/200)', 'color': COLORS['strategy4']},
]

# LinkedIn-optimized table style
LINKEDIN_TABLE_STYLE = {
    'header': {
        'fill_color': COLORS['linkedin_dark'],
        'align': 'center',
        'font': {'size': 14, 'color': 'white', 'family': THEME['font_family']},
        'height': 40,
    },
    'cells': {
        'fill_color': [[COLORS['linkedin_light'], COLORS['light_bg']] * 50],
        'align': ['left', 'right'],
        'font': {'size': 12, 'color': COLORS['text'], 'family': THEME['font_family']},
        'height': 30,
        'line': {'color': COLORS['linkedin_accent'], 'width': 1},
    },
}

# Comparison table HTML style
TABLE_HTML_STYLE = """
<style>
.metrics-table {
    width: 100%;
    border-collapse: collapse;
    font-family: Arial, sans-serif;
    font-size: 14px;
    text-align: left;
    margin-bottom: 20px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    border-radius: 5px;
    overflow: hidden;
}
.metrics-table th {
    background-color: #e9ecef;
    color: #343a40;
    padding: 12px 15px;
    border-bottom: 1px solid #dee2e6;
    border-left: none;
    border-right: none;
    border-top: none;
    font-weight: bold;
}
.metrics-table td {
    padding: 10px 15px;
    border-bottom: 1px solid #e9ecef;
    border-left: none;
    border-right: none;
    border-top: none;
}
.metrics-table tr:nth-child(even) {
    background-color: #f8f9fa;
}
.metrics-table tr:hover {
    background-color: #e9ecef;
}
.strategy-name {
    font-weight: bold;
}
.positive-value {
    color: #06d6a0;
    font-weight: 500;
}
.negative-value {
    color: #e63946;
    font-weight: 500;
}
</style>
"""

# LinkedIn-optimized table style
LINKEDIN_TABLE_HTML_STYLE = """
<style>
.metrics-table {
    width: 100%;
    border-collapse: collapse;
    font-family: Arial, sans-serif;
    font-size: 14px;
    text-align: left;
    margin-bottom: 20px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    border-radius: 8px;
    overflow: hidden;
}
.metrics-table th {
    background-color: #0077b5;
    color: white;
    padding: 14px 16px;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 13px;
    letter-spacing: 0.5px;
}
.metrics-table td {
    padding: 12px 16px;
    border-bottom: 1px solid #e9ecef;
}
.metrics-table tr:nth-child(even) {
    background-color: #f3f6f8;
}
.metrics-table tr:hover {
    background-color: #eef3f8;
}
.metrics-table tr:last-child td {
    border-bottom: none;
}
.strategy-name {
    font-weight: 600;
}
.positive-value {
    color: #057642;
    font-weight: 600;
}
.negative-value {
    color: #c91b1b;
    font-weight: 600;
}
.call-to-action {
    background-color: #f3f6f8;
    border: 1px solid #0077b5;
    border-radius: 6px;
    padding: 12px 16px;
    margin-top: 20px;
    font-weight: 600;
    color: #0077b5;
    text-align: center;
}
</style>
"""


def get_strategy_color(short_window, long_window):
    """Get a consistent color for a strategy based on its parameters"""
    key = f"{short_window}_{long_window}"
    if key in STRATEGY_COLORS:
        return STRATEGY_COLORS[key]

    # If not in predefined colors, return a default
    return COLORS['strategy5']


def apply_theme_to_figure(fig):
    """Apply the theme settings to a plotly figure"""
    fig.update_layout(
        plot_bgcolor=LAYOUT['plot_bgcolor'],
        paper_bgcolor=LAYOUT['paper_bgcolor'],
        font=LAYOUT['font'],
        margin=LAYOUT['margin'],
        hovermode=LAYOUT['hovermode'],
        legend=LAYOUT['legend'],
    )

    # Apply to all x and y axes
    fig.update_xaxes(
        gridcolor=LAYOUT['xaxis']['gridcolor'],
        zerolinecolor=LAYOUT['xaxis']['zerolinecolor'],
        title_font=LAYOUT['xaxis']['title']['font'],
        tickfont=LAYOUT['xaxis']['tickfont'],
    )

    fig.update_yaxes(
        gridcolor=LAYOUT['yaxis']['gridcolor'],
        zerolinecolor=LAYOUT['yaxis']['zerolinecolor'],
        title_font=LAYOUT['yaxis']['title']['font'],
        tickfont=LAYOUT['yaxis']['tickfont'],
    )

    return fig


def format_button_bar(fig):
    """Apply consistent styling to animation button bar"""
    if 'updatemenus' in fig.layout and fig.layout.updatemenus:
        for menu in fig.layout.updatemenus:
            for button in menu.buttons:
                button.update(
                )
    return fig


def format_slider(fig):
    """Apply consistent styling to animation slider"""
    if 'sliders' in fig.layout and fig.layout.sliders:
        for slider in fig.layout.sliders:
            slider.update(
                bgcolor=SLIDER_STYLE['bgcolor'],
                bordercolor=SLIDER_STYLE['bordercolor'],
                borderwidth=SLIDER_STYLE['borderwidth'],
                tickwidth=SLIDER_STYLE['tickwidth'],
                tickcolor=SLIDER_STYLE['tickcolor'],
                # Corrected property name
                minorticklen=SLIDER_STYLE['minorticklen'],
                currentvalue=SLIDER_STYLE['currentvalue'],
            )
    return fig


def create_watermark(fig):
    """Add a subtle watermark with LinkedIn sharing call-to-action"""
    fig.add_annotation(
        x=1.0,
        y=0,
        xref="paper",
        yref="paper",
        text="Created with MA Crossover Strategy Tool • Share on LinkedIn",
        showarrow=False,
        font=dict(
            family=THEME['font_family'],
            size=10,
            color='rgba(0,0,0,0.3)'
        ),
        xanchor="right",
        yanchor="bottom",
    )
    return fig


def create_linkedin_watermark(fig):
    """Add a LinkedIn-styled watermark with sharing call-to-action"""
    fig.add_annotation(
        x=1.0,
        y=-0.05,
        xref="paper",
        yref="paper",
        text="Created with Moving Average Crossover Strategy Tool<br>Share your analysis on LinkedIn #AlgoTrading",
        showarrow=False,
        font=dict(
            family=THEME['font_family'],
            size=12,
            color=COLORS['linkedin_blue']
        ),
        xanchor="right",
        yanchor="top",
        align="right",
        bgcolor='rgba(243, 246, 248, 0.7)',
        bordercolor=COLORS['linkedin_accent'],
        borderwidth=1,
        borderpad=4,
        opacity=0.8
    )
    return fig


def format_number(value, prefix="", suffix="", decimals=2, is_percentage=False):
    """Format numbers consistently for display"""
    if is_percentage:
        formatted = f"{value:.{decimals}f}%"
    else:
        formatted = f"{value:,.{decimals}f}"

    return f"{prefix}{formatted}{suffix}"


def create_performance_badge(fig, performance_value, x_pos, y_pos, title="Performance"):
    """Add a performance metric badge to the figure"""
    is_positive = performance_value > 0
    color = COLORS['positive'] if is_positive else COLORS['negative']
    symbol = "+" if is_positive else ""

    fig.add_annotation(
        x=x_pos,
        y=y_pos,
        xref="paper",
        yref="paper",
        text=f"<b>{title}</b><br>{symbol}{performance_value:.2f}%",
        showarrow=False,
        font=dict(
            family=THEME['font_family'],
            size=14,
            color='white'
        ),
        align="center",
        bgcolor=color,
        bordercolor=color,
        borderwidth=2,
        borderpad=4,
        opacity=0.9,
        height=60,
        width=120
    )
    return fig
