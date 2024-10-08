# Copyright (c) 2024 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Alexis Plaquet, 2024

from copy import deepcopy
import plotly
import plotly.graph_objects as go
import plotly.io as pio


def layout_paper_v1(nplots=None):
    lay = dict(
        template="plotly_white",
        showlegend=True,
        legend=dict(
            yanchor="bottom",
            y=0.0,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255,255,255,1.0)",
            bordercolor="rgba(0,0,0,1.0)",
            borderwidth=1,
            font=dict(size=12, color="black"),
        ),
        margin=dict(
            l=70,
            r=70,
            b=70,
            t=70,
        ),
        xaxis=dict(
            title_font=dict(size=16, color="black"),
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
            ticks="outside",
            tickfont=dict(size=15, color="black"),
        ),
        yaxis=dict(
            title_font=dict(size=16, color="black"),
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
            ticks="outside",
            tickfont=dict(size=15, color="black"),
        ),
        title_font=dict(size=20, color="black"),
        paper_bgcolor="rgba(1.0,1.0,1.0,0.0)",
    )
    if nplots is None:
        return lay

    for i in range(nplots):
        lay[f"xaxis{i+1}"] = deepcopy(lay["xaxis"])
        lay[f"yaxis{i+1}"] = deepcopy(lay["yaxis"])
    del lay["xaxis"]
    del lay["yaxis"]
    return lay


pio.templates["plaqper1"] = go.layout.Template(layout=layout_paper_v1())
