import sys
import numpy as np

import random

import matplotlib
matplotlib.use('svg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import mitsuba
mitsuba.set_variant("scalar_rgb")

from mitsuba.core import load_dict

def plot_samples(sampler_dict, filename, grid_on=True, proj_1d=True, dim_offset=0):
    sampler = load_dict(sampler_dict)
    sample_count = sampler.sample_count()

    sampler.seed(0)
    samples = []
    for s in range(sample_count):
        # Move to the requested dimensions in the sequence
        for i in range(dim_offset):
            sampler.next_1d()

        samples.append(sampler.next_2d())

        sampler.advance()

    xx = [samples[s][0] for s in range(sample_count)]
    yy = [samples[s][1] for s in range(sample_count)]

    # Plot 1D projections
    if proj_1d:
        fig, axes = plt.subplots(nrows=2, ncols=2,
                                 figsize=(11, 11),
                                 gridspec_kw=dict(width_ratios=[0.98, 0.02], height_ratios=[0.02, 0.98]))
        plt.subplots_adjust(wspace=0.05, hspace=0.05)

        # Plot 2D samples

        axes[1][0].scatter(xx, yy)
        axes[1][0].set_xlim([0.0, 1.0])
        axes[1][0].set_ylim([0.0, 1.0])

        axes[1][0].xaxis.set_major_locator(MultipleLocator(0.2))
        axes[1][0].yaxis.set_major_locator(MultipleLocator(0.2))

        if grid_on:
            grid_delta = 1.0 / np.sqrt(sample_count)
            axes[1][0].xaxis.set_minor_locator(MultipleLocator(grid_delta))
            axes[1][0].yaxis.set_minor_locator(MultipleLocator(grid_delta))
            axes[1][0].grid(b=True, which='minor', alpha=0.3)
        else:
            axes[1][0].xaxis.set_minor_locator(MultipleLocator(0.02))
            axes[1][0].yaxis.set_minor_locator(MultipleLocator(0.02))

        # Plot 1D projections on Y axis

        axes[1][1].vlines(0.0, 0.0, 1.0)
        proj = yy #random.sample(yy, 512)
        axes[1][1].plot(np.zeros(np.shape(proj)), proj, '_', ms=20)

        axes[1][1].set_xlim(0.0, 1.0)
        axes[1][1].set_ylim(0.0, 1.0)
        axes[1][1].axis('off')

        # Plot 1D projections on X axis

        axes[0][0].hlines(0.0, 0.0, 1)
        proj = xx #random.sample(xx, 512)
        axes[0][0].plot(proj, np.zeros(np.shape(proj)), '|', ms=20)

        axes[0][0].set_xlim(0.0, 1.0)
        axes[0][0].set_ylim(0.0, 1.0)
        axes[0][0].axis('off')

        # Make the 4th plot invisible

        axes[0][1].axis('off')
    else:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

        # Plot 2D samples
        ax.scatter(xx, yy)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])

        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.xaxis.set_minor_locator(MultipleLocator(0.02))
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(MultipleLocator(0.02))

    fig.savefig(filename)


if __name__ == '__main__':
    ####################################
    # independent sampler

    plot_samples(
        sampler_dict={
            "type" : "independent",
            "sample_count" : 64,
        },
        filename="independent_64_samples_and_proj.svg",
        grid_on=False,
        proj_1d=True
    )

    plot_samples(
        sampler_dict={
            "type" : "independent",
            "sample_count" : 1024,
        },
        filename="independent_1024_samples.svg",
        grid_on=False,
        proj_1d=False
    )


    ####################################
    # stratified sampler

    plot_samples(
        sampler_dict={
            "type" : "stratified",
            "sample_count" : 64,
            "jitter" : True,
        },
        filename="stratified_64_samples_and_proj.svg",
        grid_on=True,
        proj_1d=True
    )

    plot_samples(
        sampler_dict={
            "type" : "stratified",
            "sample_count" : 1024,
        },
        filename="stratified_1024_samples.svg",
        grid_on=False,
        proj_1d=False
    )


    ####################################
    # multijitter sampler

    plot_samples(
        sampler_dict={
            "type" : "multijitter",
            "sample_count" : 64,
            "jitter" : True,
        },
        filename="multijitter_64_samples_and_proj.svg",
        grid_on=True,
        proj_1d=True
    )

    plot_samples(
        sampler_dict={
            "type" : "multijitter",
            "sample_count" : 1024,
        },
        filename="multijitter_1024_samples.svg",
        grid_on=False,
        proj_1d=False
    )


    ####################################
    # orthogonal sampler

    plot_samples(
        sampler_dict={
            "type" : "orthogonal",
            "sample_count" : 49,
            "jitter" : True,
        },
        filename="orthogonal_49_samples_and_proj.svg",
        grid_on=True,
        proj_1d=True
    )

    plot_samples(
        sampler_dict={
            "type" : "orthogonal",
            "sample_count" : 1369,
        },
        filename="orthogonal_1369_samples.svg",
        grid_on=False,
        proj_1d=False
    )

    ####################################
    # ldsampler sampler

    plot_samples(
        sampler_dict={
            "type" : "ldsampler",
            "sample_count" : 64,
        },
        filename="ldsampler_64_samples_and_proj.svg",
        grid_on=True,
        proj_1d=True
    )

    plot_samples(
        sampler_dict={
            "type" : "ldsampler",
            "sample_count" : 1024,
        },
        filename="ldsampler_1024_samples.svg",
        grid_on=False,
        proj_1d=False
    )

    plot_samples(
        sampler_dict={
            "type" : "ldsampler",
            "sample_count" : 64,
        },
        filename="ldsampler_64_samples_and_proj_dim_32.svg",
        grid_on=True,
        proj_1d=True,
        dim_offset=2
    )

    plot_samples(
        sampler_dict={
            "type" : "ldsampler",
            "sample_count" : 1024,
        },
        filename="ldsampler_1024_samples_dim_32.svg",
        grid_on=False,
        proj_1d=False,
        dim_offset=2
    )