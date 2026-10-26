"""Figure settings shared by the three experiments."""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, LogLocator, NullFormatter, StrMethodFormatter

COLORS = ('#0072B2', '#D55E00', '#009E73', '#AA4499')
LINE_STYLES = ('-', '--', ':', '-.')


def set_plot_style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 15, 'axes.labelsize': 20,
        'axes.titlesize': 18, 'axes.titleweight': 'semibold', 'axes.titlelocation': 'left',
        'axes.titlepad': 17, 'axes.labelpad': 12, 'legend.fontsize': 15.5,
        'text.color': '#20384B', 'axes.labelcolor': '#20384B',
        'xtick.color': '#435C6E', 'ytick.color': '#435C6E', 'axes.facecolor': '#F0F3F7',
        'figure.facecolor': 'white', 'savefig.facecolor': 'white',
        'axes.edgecolor': '#B2C2CF', 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'axes.axisbelow': True, 'grid.color': 'white',
        'grid.linewidth': 1.15, 'legend.facecolor': 'white',
        'legend.edgecolor': '#D7E1E9', 'legend.framealpha': .96, 'pdf.fonttype': 42})


def save_figure(fig, filename):
    filename = Path(filename)
    fig.savefig(filename)
    plt.close(fig)
    print('Saved '+filename.name, flush=True)


def plot_history(histories, filename, column=1, reference=0.,
                 ylabel=r'Residual ($R_k$)', max_iterations=None):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.subplots_adjust(left=.20, right=.96, bottom=.18, top=.96)
    for method_index, (name, runs) in enumerate(histories.items()):
        color = COLORS[method_index]
        line = LINE_STYLES[method_index]
        for run_index, history in enumerate(runs):
            if max_iterations is not None:
                history = history[(history[:, 0] >= 1) & (history[:, 0] <= max_iterations)]
            values = history[:, column].copy()-reference
            values[values <= 0] = np.nan
            ax.plot(history[:, 0], values, color=color, ls=line, lw=2.5,
                    alpha=.68 if len(runs) > 1 else .95,
                    label=name if run_index == 0 else None)
    ax.set(xlabel='Iterations, k', ylabel=ylabel, yscale='log')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=7))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.margins(x=.025, y=.08)
    ax.legend(loc='best', handlelength=2.4)
    save_figure(fig, filename)


def plot_residual(histories, filename):
    plot_history(histories, filename)


def plot_steps(step_histories, filename, max_iterations=1000):
    set_plot_style()
    methods = ('AE-GRAAL-A', 'AE-GRAAL-B', 'aGRAAL')
    fig, axes = plt.subplots(3, 1, figsize=(6, 5), sharex=True, sharey=True)
    fig.subplots_adjust(left=.20, right=.96, bottom=.18, top=.85, hspace=.20)
    for ax, name in zip(axes, methods):
        step_history = np.asarray(step_histories[name], dtype=float).reshape(-1, 3)
        step_history = step_history[(step_history[:, 0] >= 1) & (step_history[:, 0] <= max_iterations)]
        for column, color, line, label in [
                (1, COLORS[0], '-', r'$\lambda_k$'),
                (2, COLORS[1], '--', r'$1/L_k$')]:
            values = step_history[:, column]
            values = np.where(np.isfinite(values) & (values > 0), values, np.nan)
            ax.plot(step_history[:, 0], values, color=color, ls=line, lw=2.5, label=label)
        ax.set(yscale='log', xlim=(0, max_iterations))
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=3))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.margins(y=.08)
        ax.text(.98, .90, name, transform=ax.transAxes, ha='right', va='top',
                fontsize=plt.rcParams['legend.fontsize'],
                bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=2))
    axes[-1].set_xlabel('Iterations, k')
    axes[-1].xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    fig.supylabel('Step size', x=.02, fontsize=plt.rcParams['axes.labelsize'],
                  color=plt.rcParams['axes.labelcolor'])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, handlelength=2.4)
    save_figure(fig, filename)


def plot_c_scan(scenario_counts, c_values, filename, label_prefix='Scenario '):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.subplots_adjust(left=.20, right=.96, bottom=.18, top=.96)
    for scenario_index, (scenario, counts) in enumerate(scenario_counts.items()):
        medians = np.median(counts, axis=0)
        low, high = np.min(counts, axis=0), np.max(counts, axis=0)
        ax.plot(c_values, medians, color=COLORS[scenario_index], ls=LINE_STYLES[scenario_index], lw=2.5,
                marker='o', ms=4, label=f'{label_prefix}{scenario}')
        ax.fill_between(c_values, low, high, color=COLORS[scenario_index], alpha=.12)
    ax.set(xlabel='Parameter c', ylabel='Iterations, k', xlim=(1, 3))
    ax.set_xticks([1, 1.5, 2, 2.5, 3])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.margins(y=.08)
    ax.legend(loc='best', handlelength=2.4)
    save_figure(fig, filename)
