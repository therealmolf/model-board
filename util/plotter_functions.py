"""
Module containing plotter functions

"""


import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots


def usage_co2_scatter(month_df: pd.DataFrame) -> go._figure.Figure:
    """
        Create two graphs for usage and co2 emission per month.

        Arguments:
            df (pd.DataFrame): A dataframe grouped by month
                with usage and co2 columns

        Returns:
            go._figure.Figure: A plotly Figure object with two subplots.
    """

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
                    'Average Usage (kWh) per Month in 2018',
                    'Average CO2 per Month in 2018'))

    # Set the template for the figure
    fig['layout']['template'] = 'ggplot2'

    usage_graph = go.Scatter(
        x=month_df.index,
        y=month_df['Usage_kWh'],
        fill='tozeroy',
        # fillcolor='rgba(255, 0, 0, 0.5)'
        )

    co_graph = go.Scatter(
        x=month_df.index,
        y=month_df['CO2(tCO2)'],
        fill='tozeroy',
    )

    # Add the Scatter plots to the subplot object
    fig.append_trace(usage_graph, row=1, col=1)
    fig.append_trace(co_graph, row=1, col=2)

    # Adding titles to xaxis and yaxis of the subplots
    fig['layout']['xaxis'].update(title='Month')
    fig['layout']['xaxis2'].update(title="Month")
    fig['layout']['yaxis'].update(title='Usage KWh')
    fig['layout']['yaxis2'].update(title="CO2(tCO2)")

    fig['layout']['font'].update(
        family='Times New Roman',
        size=10,
        color='gray'
    )

    fig.update_layout({
        "title": "Usage and CO2 from the Steel Industry Dataset"
    })

    # Changing the fonts of the tick labels and the title
    fig['layout']['title']['font'].update(
        family='TImes New Roman',
        size=30,
        color='black',
    )

    return fig


def class_dist_bar(class_dist: pd.DataFrame) -> go._figure.Figure:
    """
        Returns a figure object representing the bar graph of the
        class distribution

        Arguments:
            class_dist (pd.DataFrame): class distribution series

        Returns:
            go._figure.Figure: Bar Graph Figure object

    """

    fig = go.Figure()

    fig['layout']['template'] = 'ggplot2'
    fig['layout']['font'].update(
        family='Times New Roman',
        size=20,
        color="gray"
    )
    fig.update_layout(
        {
            "title": "Class Distribution of Steel Industry",
            "title_font": {
                "size": 30,
                "color": "black"
            },
            "xaxis_title": "No. of Instances",
            "yaxis_title": "Load Type Class"
        }
    )

    bar = go.Bar(
        x=class_dist['count'],
        y=class_dist['Load_Type'],
        orientation='h',
        opacity=1,
        )

    fig.add_trace(bar)

    return fig
