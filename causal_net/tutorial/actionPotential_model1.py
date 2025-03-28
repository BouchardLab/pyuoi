#!/usr/bin/env python3
'''
Realistic Action Potential Simulator with argparse parameters and PNG saving

Usage:
./action_potential_simulator.py --resting_pot -70 --peak_pot 40 --threshold_pot -55 --save_path neuron_trace.png
'''
import numpy as np
import matplotlib.pyplot as plt
import argparse

def generate_action_potential(
    time_ms,
    resting_pot=-70,
    peak_pot=40,
    threshold_pot=-55,
    spike_start=5,
    depolar_dur=0.5,
    peak_dur=0.5,
    repolar_dur=1.0,
    hyperpolar_dur=3.0,
    hyperpolar_depth=10
):
    voltage_trace = np.full(time_ms.shape, resting_pot)
    t_rise_end = spike_start + depolar_dur
    t_peak_end = t_rise_end + peak_dur
    t_repol_end = t_peak_end + repolar_dur
    t_hyperpol_end = t_repol_end + hyperpolar_dur

    for i, t in enumerate(time_ms):
        if spike_start <= t < t_rise_end:
            voltage_trace[i] = resting_pot + (peak_pot - resting_pot) * (1 - np.exp(-(t - spike_start)/0.1))
        elif t_rise_end <= t < t_peak_end:
            voltage_trace[i] = peak_pot
        elif t_peak_end <= t < t_repol_end:
            voltage_trace[i] = peak_pot - (peak_pot - resting_pot) * (1 - np.exp(-(t - t_peak_end)/0.3))
        elif t_repol_end <= t < t_hyperpol_end:
            voltage_trace[i] = resting_pot - hyperpolar_depth * np.exp(-(t - t_repol_end)/1.0)
        elif t >= t_hyperpol_end:
            voltage_trace[i] = resting_pot
    return voltage_trace

def YYgenerate_action_potential(time_ms, resting_pot, peak_pot, threshold_pot,
                              spike_start, depolar_dur, peak_dur, repolar_dur,
                              hyperpolar_dur, hyperpolar_depth):
    voltage_trace = np.full(time_ms.shape, resting_pot)
    t_rise_end = spike_start + depolar_dur
    t_peak_end = t_rise_end + peak_dur
    t_repol_end = t_peak_end + repolar_dur
    t_hyperpol_end = t_repol_end + hyperpolar_dur

    for i, t in enumerate(time_ms):
        if spike_start <= t < t_rise_end:
            voltage_trace[i] = resting_pot + (peak_pot - resting_pot) * (1 - np.exp(-(t - spike_start)/0.1))
        elif t_rise_end <= t < t_peak_end:
            voltage_trace[i] = peak_pot
        elif t_peak_end <= t < t_repol_end:
            voltage_trace[i] = peak_pot - (peak_pot - resting_pot) * (1 - np.exp(-(t - t_peak_end)/0.3))
        elif t_repol_end <= t < t_hyperpol_end:
            voltage_trace[i] = resting_pot - hyperpolar_depth * np.exp(-(t - t_repol_end)/1.0)
        elif t >= t_hyperpol_end:
            voltage_trace[i] = resting_pot
    return voltage_trace

def plot_action_potential(time_ms, voltage_trace, threshold_pot, save_path):
    plt.figure(figsize=(12, 4))
    plt.plot(time_ms, voltage_trace, label='Action Potential')
    plt.axhline(y=threshold_pot, color='gray', linestyle='--', label=f'Threshold Potential ({threshold_pot} mV)')
    plt.title("Realistic Action Potential of Excitatory Neuron")
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane Potential (mV)")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, format='png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate realistic action potential trace.")
    parser.add_argument('--resting_pot', type=float, default=-70, help='Resting potential (mV)')
    parser.add_argument('--peak_pot', type=float, default=40, help='Peak potential (mV)')
    parser.add_argument('--threshold_pot', type=float, default=-55, help='Threshold potential (mV)')
    parser.add_argument('--spike_start', type=float, default=5, help='Spike start time (ms)')
    parser.add_argument('--depolar_dur', type=float, default=0.5, help='Depolarization duration (ms)')
    parser.add_argument('--peak_dur', type=float, default=0.5, help='Peak duration (ms)')
    parser.add_argument('--repolar_dur', type=float, default=1.0, help='Repolarization duration (ms)')
    parser.add_argument('--hyperpolar_dur', type=float, default=3.0, help='Hyperpolarization duration (ms)')
    parser.add_argument('--hyperpolar_depth', type=float, default=10, help='Hyperpolarization depth below resting potential (mV)')
    parser.add_argument('--total_time', type=float, default=20.0, help='Total simulation time (ms)')
    parser.add_argument('--time_resolution', type=float, default=0.1, help='Time resolution (ms)')
    parser.add_argument('--save_path', type=str, default='out/action_potential.png', help='Path to save PNG plot')
    
    args = parser.parse_args()

    time_ms = np.arange(0, args.total_time + args.time_resolution, args.time_resolution)
    voltage_trace = generate_action_potential(
        time_ms,
        args.resting_pot,
        args.peak_pot,
        args.threshold_pot,
        args.spike_start,
        args.depolar_dur,
        args.peak_dur,
        args.repolar_dur,
        args.hyperpolar_dur,
        args.hyperpolar_depth
    )
    #voltage_trace = generate_action_potential(time_ms)

    plot_action_potential(time_ms, voltage_trace, args.threshold_pot, args.save_path)

if __name__ == "__main__":
    main()
