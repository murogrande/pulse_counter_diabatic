import copy
import numpy as np
import torch
import matplotlib.pyplot as plt

import pulser
from pulser.waveforms import InterpolatedWaveform, CustomWaveform

from pulse_counter_diabatic.counter_diabatic import CounterDiabaticPulse

from pulser import AnalogDevice
import sys
import json
import networkx as nx

from pulser import Register
import numpy as np
import networkx as nx
from pulser.register.special_layouts import TriangularLatticeLayout


# ── DMRG ground-state helper ─────────────────────────────────────────────────
from emu_mps import (
    MPSBackend, MPSConfig, Solver, StateResult,
    Fidelity, Occupation, CorrelationMatrix, EntanglementEntropy, BitStrings
)

def dmrg_ground_state(seq):
    cfg = MPSConfig(with_modulation=False, dt=10, solver=Solver.DMRG, observables=[StateResult(evaluation_times=[1.0])])
    backend = MPSBackend(seq,config=cfg)
    result= backend.run() 
    target = result.final_state
    return target


def create_lattice_with_holes(
    N: int,
    density: float,
    spacing: float = 5.0,
    n_traps: int = 200,
) -> tuple:

    if not (0 < density <= 1.0):
        raise ValueError(f"density must be in (0, 1], got {density}")

    L = int(np.ceil(N / density))
    if L > n_traps:
        raise ValueError(
            f"Need {L} candidate sites but layout only has {n_traps} traps. "
            "Increase n_traps or decrease N/density."
        )

    reg_layout = TriangularLatticeLayout(n_traps, spacing)
    coords = reg_layout.coords

    idx_sorted = np.argsort(np.linalg.norm(coords, axis=1))
    idx_avail  = idx_sorted[:L]

    idx_list = np.random.choice(idx_avail, size=N, replace=False).tolist()

    register = reg_layout.define_register(*idx_list)

    selected_coords = coords[idx_list]
    G = nx.Graph()
    G.add_nodes_from(range(N))
    threshold = spacing * 1.05
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(selected_coords[i] - selected_coords[j]) < threshold:
                G.add_edge(i, j)

    return register, G, idx_list

nruns=10

all_benchmark_data = []
N        = int(sys.argv[1])
density  = float(sys.argv[2])
T        = int(sys.argv[3])
nfourier = int(sys.argv[4])

for run in range(nruns):
    reg, graph, idx_list = create_lattice_with_holes(
        N       = N,
        density = density,
    )
    
    # --- Baseline adiabatic protocol ------------------------------------------------      
    dt = 10   
    delta_max = AnalogDevice.channels['rydberg_global'].max_abs_detuning
    omega_max = AnalogDevice.channels['rydberg_global'].max_amp
    C6= AnalogDevice.interaction_coeff

    pulse_times = np.arange(T)                              
    omega_samples = omega_max * np.sin(0.5 * np.pi * np.sin(np.pi * pulse_times / T)) ** 2
    omega_wf = CustomWaveform(omega_samples)
    adiabatic_pulse = pulser.Pulse(
        omega_wf,         # Ω(t) = 0
        InterpolatedWaveform(T, [-delta_max, delta_max]),         # δ ramp
        0.0,
    )

    seq = pulser.Sequence(reg, pulser.MockDevice)
    seq.declare_channel("ising_global", "rydberg_global")
    seq.add(adiabatic_pulse, "ising_global")

    target=dmrg_ground_state(seq)


    config_for_cd = MPSConfig(dt=dt, observables=[])

    # LCD only
    cd_lcd = CounterDiabaticPulse(seq, config_for_cd)
    seq_data_lcd = cd_lcd.solver(cold_fourier=0)

    # COLD 
    cd_cold = CounterDiabaticPulse(seq, config_for_cd)
    seq_data_cold = cd_cold.solver(cold_fourier=nfourier, nruns=100, lr=1e-2)


    config = MPSConfig(
        dt=10,
        observables=[Fidelity(evaluation_times=[1.0], state=target),StateResult(evaluation_times=[1.0])])

    print("Baseline schedule...")
    backend = MPSBackend(seq, config=config)
    results_baseline = backend.run()
    print("Emulating LCD schedule...")
    results_lcd=MPSBackend._run_from_sequence_data(seq_data_lcd, config)
    print("Emulating COLD schedule...")
    results_cold=MPSBackend._run_from_sequence_data(seq_data_cold, config)

    fid_baseline = results_baseline.get_tagged_results()['fidelity']
    fid_lcd  = results_lcd.get_tagged_results()['fidelity']
    fid_cold = results_cold.get_tagged_results()['fidelity']

    print(f"Baseline  ground-state fidelity: {fid_baseline}")
    print(f"LCD  ground-state fidelity: {fid_lcd}")
    print(f"COLD ground-state fidelity: {fid_cold}")

    run_data = {
        "run_id": run,
        "fidelities": {
            "baseline": fid_baseline[0].item(),
            "lcd": fid_lcd[0].item(),
            "cold": fid_cold[0].item()
        },
    }
    
    # 3. Append to our main list
    all_benchmark_data.append(run_data)

# 4. Save everything to a JSON file after the loop finishes

filename = "./benchmark_results/benchmark_results_nfourier_"+str(nfourier)+"_dt_"+str(dt)+"_N_"+str(N)+"_density_"+str(density)+".json"
with open(filename, "w") as f:
    json.dump(all_benchmark_data, f, indent=4)
    
print(f"\nAll benchmark data successfully saved to {filename}")