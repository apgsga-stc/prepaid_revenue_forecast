import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from statsmodels.tsa.seasonal import STL
from pathlib import Path

########################################################################################################################

def stl_fitting(observation_df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    TimeSeries Decomposition with LOESS and SARIMA

    :param observation_df: TimeSeries Dataframe
    :return: Dataframe with STL-decomposition
    """
    stl = STL(
        # endog=sunrise_df.rev_net_total_per_day,
        endog=observation_df[target_column],
        period=12,
        robust=True,  # Flag indicating whether to use a weighted version that is robust to some forms of outliers.
        # For Anomally detection set to "True"
    )
    res = stl.fit()

    # Shoving everything into a Dataframe for ease of handling
    res_df = pd.DataFrame(
        {
            "observed": res.observed,
            "trend": res.trend,
            "seasonal": res.seasonal,
            "residuals": res.resid,
            "weights": res.weights,
        }
    )

    return res_df

########################################################################################################################

def plot_stl_fitting(res_df:pd.DataFrame , brand_name: str, ANALYSIS_DIR)  -> None:
    """

    Create html-file with TimeSeries-Plots as well its decompositions and Residual-Analysis

    :param res_df: STL-fitted Dataframe
    :param brand_name: Brand Name (Sunrise, Yallo, Lebara)
    :param ANALYSIS_DIR: Directory, where html-files ends up.
    :return: None
    """
    fig = make_subplots(
        rows=3,
        cols=2,
        column_widths=[0.6, 0.4],
        shared_xaxes=True,
        # shared_yaxes=True,
        vertical_spacing=0.005,
        horizontal_spacing=0.05,
        # subplot_titles=("observed", "trend", "seasonal", "residuals")
    )


    fig.add_trace(
        go.Bar(
            x=res_df.index,
            y=res_df.observed,
            name="Observations",
            # mode="markers",
            opacity=0.5,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=res_df.index,
            y=res_df.trend,
            name="Trend (LOESS)",
            mode="lines",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=res_df.index,
            y=res_df.trend + res_df.seasonal,
            name="Estimation (Trend+Seasonality)",
            mode="lines+markers",
        ),
        row=1,
        col=1,
    )


    fig.add_trace(
        go.Scatter(
            x=res_df.index,
            y=res_df.seasonal,
            name="Seasonality (SARIMA)",
            mode="lines+markers",
            line_shape="spline",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=res_df.index,
            y=res_df.residuals,
            name="Residuals (CHF)",
            visible="legendonly",
        ),
        row=3,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=res_df.index,
            y=res_df.residuals / res_df.observed * 100,
            name="Residuals (%)",
        ),
        row=3,
        col=1,
    )

    # for error in [lower_band, upper_band]:
    #    fig.add_hline(
    #        y=error,
    #        line_dash="dot",
    #        row=3,
    #        col="all",
    #        line=dict(color="red"),
    #        # annotation_text="Jan 1, 2018 baseline",
    #        # annotation_position="bottom right",
    #    )

    #########
    residuals_percentage = res_df.residuals / res_df.observed * 100
    fig.add_trace(
        go.Histogram(
            x=residuals_percentage,
            xbins=dict(size=0.2),  # start=-3.0, #end=4,
            opacity=0.5,
            name="Histogram: Residuals (%)",
            showlegend=False,
            marker_color="grey",
        ),
        row=2,
        col=2,
    )

    fig.add_trace(
        go.Box(
            x=residuals_percentage,
            boxpoints="all",
            jitter=0.4,
            notched=True,
            name="Boxplot",
            showlegend=False,
            boxmean="sd",
            marker_color="grey",
        ),
        row=3,
        col=2,
    )

    ##########

    _date_vlines_ = (
        pd.Series(
            pd.date_range(
                start=res_df.index.min() - pd.DateOffset(years=1),
                end=res_df.index.max() + pd.DateOffset(years=1),
                freq="Y",
            )
        )
        - pd.DateOffset(days=15)
    )

    for _i_ in [1, 2, 3]:
        for _x_ in _date_vlines_:
            fig.add_vline(
                x=_x_,
                line_dash="dot",
                row=_i_,
                col=1,
                line=dict(color="grey"),
                # annotation_text="Jan 1, 2018 baseline",
                # annotation_position="bottom right",
            )

    fig.update_layout(
        title_text=f"{brand_name} Prepaid: Anomaly Detection via Seasonality-Trend Decomposition using SARIMA & LOESS",
        margin=dict(l=20, r=20, t=30, b=20),
        hovermode="x unified",
        legend=dict(x=0.85, y=0.95, bgcolor="white"),
    )

    _link_to_file_ = ANALYSIS_DIR / f"FIA0590_{brand_name}_decomposition.html"
    fig.write_html(_link_to_file_)

    print(_link_to_file_)

########################################################################################################################


def plot_stacked_timeseries(plot_df: pd.DataFrame, ANALYSIS_DIR) -> None:
    """
    Explanations...
    """

    # plot_df = prepaid_df
    _brands_ = plot_df.Brand_Name.drop_duplicates()

    _colors_ = ["red", "orange", "blue", "green"]

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        vertical_spacing=0.005,
        horizontal_spacing=0.01,
        subplot_titles=(
            "Actual Net Revenue Per Month",
            "Adjusted monthly Net Revenue Per 30 days",
        ),
    )

    for column in zip(_brands_, _colors_):
        subplot_df = plot_df[plot_df.Brand_Name == column[0]]

        fig.add_trace(
            go.Scatter(
                x=subplot_df.Calendar_Year_Month_Date,
                y=subplot_df.rev_net_total,
                name=column[0],
                stackgroup="one",
                line=dict(color=column[1]),
                # groupnorm="percent",
                legendgroup=column[0],
                # marker=dict(color=colors[column]),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=subplot_df.Calendar_Year_Month_Date,
                y=subplot_df.rev_net_total_per_day * 30,
                name=column[0],
                stackgroup="one",
                # groupnorm="percent",
                legendgroup=column[0],
                showlegend=False,
                line=dict(color=column[1]),
                # marker=dict(color=colors[column]),
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        title_text="Stacked Prepaid Net Revenue per Month",
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="x unified",
    )
    _link_to_file_ = ANALYSIS_DIR / "FIA0590_stacked_revenue.html"
    fig.write_html(_link_to_file_)

    print(_link_to_file_)