import copy
import numpy as np
import torch
import matplotlib.pyplot as plt

import pulser
from pulser.waveforms import InterpolatedWaveform, CustomWaveform

import emu_sv
from emu_sv import SVConfig, SVBackend, Occupation, StateResult, StateVector, Fidelity

from pulse_counter_diabatic.counter_diabatic import CounterDiabaticPulse

from pulser import AnalogDevice
import sys
import json
import networkx as nx

from pulser import Register

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


import numpy as np
import networkx as nx
from pulser import Register

def create_lattice_clusters_and_chains(
    n_clusters: int = 3,
    atoms_per_cluster: int = 7,  # Target number of atoms in the final cluster
    atoms_per_chain: int = 3,
    fill_fraction: float = 0.7,  # Proportion of sites kept (e.g., 70%)
    spacing: float = 5.0,
    seed: int | None = None
):
    """
    Generates a Pulser register of interconnected atomic clusters on a triangular lattice.
    Clusters are generated with random holes based on the fill_fraction.
    """
    rng = np.random.default_rng(seed)

    # The 6 nearest-neighbor directions on a triangular lattice grid (u, v)
    DIRS = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]

    def to_cartesian(u, v):
        """Converts discrete lattice coordinates to continuous XY plane."""
        x = (u + 0.5 * v) * spacing
        y = (v * np.sqrt(3) / 2) * spacing
        return x, y

    def get_base_patch(n_sites):
        """Creates a dense, roughly circular patch of lattice sites."""
        radius = int(np.sqrt(n_sites)) + 3
        candidates = []
        for du in range(-radius, radius + 1):
            for dv in range(-radius, radius + 1):
                x, y = to_cartesian(du, dv)
                candidates.append((x**2 + y**2, du, dv))
        candidates.sort() # Sort by distance from origin
        return [(u, v) for _, u, v in candidates[:n_sites]]

    # 1. Determine the size of the dense template needed
    dense_sites_needed = int(atoms_per_cluster / fill_fraction)
    dense_template = get_base_patch(dense_sites_needed)
    
    def sample_cluster():
        """Returns a cluster with holes by sampling the dense template."""
        indices = rng.choice(len(dense_template), size=atoms_per_cluster, replace=False)
        return [dense_template[i] for i in indices]

    # Initialize the first cluster
    c0 = sample_cluster()
    occupied = set(c0)
    clusters = [c0]
    chains = []

    # Iteratively attach new clusters
    for i in range(1, n_clusters):
        placed = False
        parents = list(range(i))
        rng.shuffle(parents) # Randomize which cluster we branch from

        for p in parents:
            if placed: break
            dirs = list(DIRS)
            rng.shuffle(dirs)

            for (du, dv) in dirs:
                parent_patch = clusters[p]
                
                # Projection function to find the furthest atom in a given direction
                dx, dy = to_cartesian(du, dv)
                def proj(u, v):
                    px, py = to_cartesian(u, v)
                    return px * dx + py * dy
                
                # Attachment atom on the parent (furthest in direction (du, dv))
                attach_p = max(parent_patch, key=lambda node: proj(node[0], node[1]))

                # Build the chain nodes
                chain_nodes = []
                for k in range(1, atoms_per_chain + 1):
                    chain_nodes.append((attach_p[0] + k * du, attach_p[1] + k * dv))
                
                chain_end = chain_nodes[-1] if chain_nodes else attach_p
                target_attach_c = (chain_end[0] + du, chain_end[1] + dv)

                # Sample a fresh child cluster with holes
                child_base = sample_cluster()

                # Attachment point on the child (furthest in OPPOSITE direction)
                attach_c_base = max(child_base, key=lambda node: proj(-node[0], -node[1]))

                # Shift the child cluster so its attachment point lands exactly on target
                shift_u = target_attach_c[0] - attach_c_base[0]
                shift_v = target_attach_c[1] - attach_c_base[1]
                shifted_patch = [(u + shift_u, v + shift_v) for u, v in child_base]

                # Collision check on the integer grid
                new_nodes = set(chain_nodes) | set(shifted_patch)
                if occupied.isdisjoint(new_nodes):
                    occupied.update(new_nodes)
                    chains.append(chain_nodes)
                    clusters.append(shifted_patch)
                    placed = True
                    break

        if not placed:
            raise RuntimeError("Could not place cluster! The layout became trapped. Try a different seed.")

    # Convert everything to cartesian coordinates for Pulser & NetworkX
    coords = {}
    node_idx = 0
    cluster_ids = []
    
    for cl in clusters:
        ids = []
        for (u, v) in cl:
            qid = f"q{node_idx}"
            coords[qid] = to_cartesian(u, v)
            ids.append(qid)
            node_idx += 1
        cluster_ids.append(ids)

    chain_ids = []
    for ch in chains:
        ids = []
        for (u, v) in ch:
            qid = f"q{node_idx}"
            coords[qid] = to_cartesian(u, v)
            ids.append(qid)
            node_idx += 1
        chain_ids.append(ids)

    reg = Register(coords)

    # Build NetworkX Graph (edges connect atoms separated by exactly `spacing`)
    G = nx.Graph()
    G.add_nodes_from(coords.keys())
    qids = list(coords.keys())
    for i in range(len(qids)):
        for j in range(i + 1, len(qids)):
            p1, p2 = np.array(coords[qids[i]]), np.array(coords[qids[j]])
            if np.linalg.norm(p1 - p2) < spacing * 1.05:
                G.add_edge(qids[i], qids[j])

    return reg, G, cluster_ids, chain_ids

nruns=10

all_benchmark_data = []
n_clusters=int(sys.argv[1])
atoms_per_cluster=int(sys.argv[2])
atoms_per_chain=int(sys.argv[3])
    
for run in range(nruns):

    reg, graph, cluster_ids, chain_ids = create_lattice_clusters_and_chains(
    n_clusters=n_clusters, 
    atoms_per_cluster=atoms_per_cluster, 
    atoms_per_chain=atoms_per_chain, 
    spacing=7.0,
    )
    
    # --- Baseline adiabatic protocol ------------------------------------------------
    T = 1000        
    dt = 1           
    delta_max = 10.0  
    omega_max = 10.0   
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
    seq_data_cold = cd_cold.solver(cold_fourier=5, nruns=100, lr=1e-2)


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

filename = "benchmark_results_long_nc"+str(n_clusters)+"_na"+str(atoms_per_cluster)+"_nac"+str(atoms_per_chain)+".json"
with open(filename, "w") as f:
    json.dump(all_benchmark_data, f, indent=4)
    
print(f"\nAll benchmark data successfully saved to {filename}")