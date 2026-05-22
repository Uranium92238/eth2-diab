#!/usr/bin/env python3
"""
Plot population and coherence means from bootstrap analysis in a combined figure.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Set font to Times
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'

# Read population data
with open('population_boot.dat', 'r') as f:
    pop_header_line = f.readline().strip('#').strip()
    
pop_data = pd.read_csv('population_boot.dat', sep=r'\s+', comment='#', names=pop_header_line.split())

# Read coherence data - Define which coherences to plot
# Comment out any coherence pair you don't want to plot
COHERENCE_PAIRS = [
    # (1, 2),
    # (1, 3),
    # (1, 4),
    # (1, 5),
    # (1, 6),
    (2, 3),
    # (2, 4),
    # (2, 5),
    # (2, 6),
    (3, 4),
    # (3, 5),
    # (3, 6),
    (4, 5),
    (4, 6),
    # (5, 6),
]

# Load coherence data for all pairs
coherence_data = {}
for i, j in COHERENCE_PAIRS:
    filename = f'coherence_{i}_{j}.dat'
    with open(filename, 'r') as f:
        header_line = f.readline().strip('#').strip()
    coh_df = pd.read_csv(filename, sep=r'\s+', comment='#', names=header_line.split())
    coherence_data[(i, j)] = {
        'time': coh_df['time'],
        'mean': coh_df[f'Coh{i}.{j}_mean'],
        'ci_low': coh_df[f'Coh{i}.{j}_ci_low'],
        'ci_high': coh_df[f'Coh{i}.{j}_ci_high']
    }


# Extract population data
time_pop = pop_data['time']
pop1_mean = pop_data['Pop1_mean']
pop2_mean = pop_data['Pop2_mean']
pop3_mean = pop_data['Pop3_mean']
pop4_mean = pop_data['Pop4_mean']
pop5_mean = pop_data['Pop5_mean']
pop6_mean = pop_data['Pop6_mean']

pop1_ci_low = pop_data['Pop1_ci_low']
pop1_ci_high = pop_data['Pop1_ci_high']
pop2_ci_low = pop_data['Pop2_ci_low']
pop2_ci_high = pop_data['Pop2_ci_high']
pop3_ci_low = pop_data['Pop3_ci_low']
pop3_ci_high = pop_data['Pop3_ci_high']
pop4_ci_low = pop_data['Pop4_ci_low']
pop4_ci_high = pop_data['Pop4_ci_high']
pop5_ci_low = pop_data['Pop5_ci_low']
pop5_ci_high = pop_data['Pop5_ci_high']
pop6_ci_low = pop_data['Pop6_ci_low']
pop6_ci_high = pop_data['Pop6_ci_high']


# Toggle confidence intervals and coherence plotting on/off
SHOW_CI = False  # Set to False to hide confidence intervals
SHOW_COHERENCE = True  # Set to False to hide coherence subplot in combined figure

# Create figure with two subplots (2 rows, 1 column)
if SHOW_COHERENCE:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
else:
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

# ========== Top subplot: Populations ==========
# S0 population
# ax1.plot(time_pop, pop1_mean, label=r'$S_0$', linewidth=2)
# if SHOW_CI:
#     ax1.fill_between(time_pop, pop1_ci_low, pop1_ci_high, alpha=0.2)

# S1 population
ax1.plot(time_pop, pop2_mean, label=r'$S_1$', linewidth=2, color='red')
if SHOW_CI:
    ax1.fill_between(time_pop, pop2_ci_low, pop2_ci_high, alpha=0.2, color='red')

# S2 population
ax1.plot(time_pop, pop3_mean, label=r'$S_2$', linewidth=2, color='blue')
if SHOW_CI:
    ax1.fill_between(time_pop, pop3_ci_low, pop3_ci_high, alpha=0.2, color='blue')

# S3 population (commented out by default)
ax1.plot(time_pop, pop4_mean, label=r'$S_3$', linewidth=2, color='green')
if SHOW_CI:
    ax1.fill_between(time_pop, pop4_ci_low, pop4_ci_high, alpha=0.2, color='green')

# S4 population (commented out by default)
ax1.plot(time_pop, pop5_mean, label=r'$S_4$', linewidth=2, color='purple')
if SHOW_CI:
    ax1.fill_between(time_pop, pop5_ci_low, pop5_ci_high, alpha=0.2, color='purple')

# S5 population (commented out by default)
ax1.plot(time_pop, pop6_mean, label=r'$S_5$', linewidth=2, color='orange')
if SHOW_CI:
    ax1.fill_between(time_pop, pop6_ci_low, pop6_ci_high, alpha=0.2, color='orange')

ax1.set_xlabel(r'Time (fs)', fontsize=16)
ax1.set_ylabel(r'Population', fontsize=16)
ax1.set_title(r'State Populations vs Time', fontsize=18, fontweight='bold')
ax1.legend(loc='best', fontsize=14)
ax1.grid(False)
ax1.set_xlim(0, 500)
ax1.set_ylim(0, 1.05)

# ========== Bottom subplot: Coherence ==========
if SHOW_COHERENCE:
    for i, j in COHERENCE_PAIRS:
        coh = coherence_data[(i, j)]
        ax2.plot(coh['time'], coh['mean'], label=rf'$|\rho_{{{i}{j}}}|$', linewidth=2)
        if SHOW_CI:
            ax2.fill_between(coh['time'], coh['ci_low'], coh['ci_high'], alpha=0.2)

    ax2.set_xlabel(r'Time (fs)', fontsize=16)
    ax2.set_ylabel(r'Coherence', fontsize=16)
    ax2.set_title(r'Coherence $|\rho_{ij}|$ vs Time', fontsize=18, fontweight='bold')
    ax2.legend(loc='best', fontsize=18, ncol=2)
    ax2.grid(False)
    ax2.set_xlim(0, 500)
    # ax2.set_ylim(0, 0.1)

# Adjust layout and save combined figure
plt.tight_layout()
plt.savefig('tahir.pdf', bbox_inches='tight')
print("Combined plot saved as 'tahir.pdf'")
plt.show()

# ========== Separate Population Figure ==========
fig_pop, ax_pop = plt.subplots(1, 1, figsize=(10, 6))

# S1 population
ax_pop.plot(time_pop, pop2_mean, label=r'$S_1$', linewidth=2, color='red')
if SHOW_CI:
    ax_pop.fill_between(time_pop, pop2_ci_low, pop2_ci_high, alpha=0.2, color='red')

# S2 population
ax_pop.plot(time_pop, pop3_mean, label=r'$S_2$', linewidth=2, color='blue')
if SHOW_CI:
    ax_pop.fill_between(time_pop, pop3_ci_low, pop3_ci_high, alpha=0.2, color='blue')

# S3 population 
ax_pop.plot(time_pop, pop4_mean, label=r'$S_3$', linewidth=2, color='green')
if SHOW_CI:
    ax_pop.fill_between(time_pop, pop4_ci_low, pop4_ci_high, alpha=0.2, color='green')

# S4 population 
ax_pop.plot(time_pop, pop5_mean, label=r'$S_4$', linewidth=2, color='purple')
if SHOW_CI:
    ax_pop.fill_between(time_pop, pop5_ci_low, pop5_ci_high, alpha=0.2, color='purple')

# S5 population 
ax_pop.plot(time_pop, pop6_mean, label=r'$S_5$', linewidth=2, color='orange')
if SHOW_CI:
    ax_pop.fill_between(time_pop, pop6_ci_low, pop6_ci_high, alpha=0.2, color='orange')

ax_pop.set_xlabel(r'Time (fs)', fontsize=16)
ax_pop.set_ylabel(r'Population', fontsize=16)
ax_pop.set_title(r'State Populations vs Time', fontsize=18, fontweight='bold')
ax_pop.legend(loc='best', fontsize=14)
ax_pop.grid(False)
ax_pop.set_xlim(0, 500)
ax_pop.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig('tahir_pop.pdf', bbox_inches='tight')
print("Population plot saved as 'tahir_pop.pdf'")
plt.show()

# ========== Separate Coherence Figure ==========
fig_coh, ax_coh = plt.subplots(1, 1, figsize=(10, 6))

for i, j in COHERENCE_PAIRS:
    coh = coherence_data[(i, j)]
    ax_coh.plot(coh['time'], coh['mean'], label=rf'$|\rho_{{{i}{j}}}|$', linewidth=2)
    if SHOW_CI:
        ax_coh.fill_between(coh['time'], coh['ci_low'], coh['ci_high'], alpha=0.2)

ax_coh.set_xlabel(r'Time (fs)', fontsize=16)
ax_coh.set_ylabel(r'Coherence', fontsize=16)
ax_coh.set_title(r'Coherence $|\rho_{ij}|$ vs Time', fontsize=18, fontweight='bold')
ax_coh.legend(loc='best', fontsize=18, ncol=2)
ax_coh.grid(False)
ax_coh.set_xlim(0, 500)
# ax_coh.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('tahir_coh.pdf', bbox_inches='tight')
print("Coherence plot saved as 'tahir_coh.pdf'")
plt.show()
