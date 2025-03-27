"""
Plot EDP comparison across different architectures for a given model configuration.
"""

import os
import sys
import pickle
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

sys.path.append(os.getcwd())
from src.config import LLAMA_3_8B, LLMConfig
from src.util import CME_T, Stage

def get_cmes_full_model_all_layers(cmes: list[CME_T], model: LLMConfig, stage: Stage = Stage.PREFILL):
    """Generalize the zigzag results (for single layers) to a full LLM
    @param prefill: whether the results are from a prefill or decode phase simulation"""

    number_of_runs = 1 if stage == Stage.PREFILL else model.decode_size
    return [cme * model.get_post_simulation_multiplier(cme.layer.name) * number_of_runs for cme in cmes]

def get_total_edp(cmes: List[dict], model_name: str, stage: Stage) -> Tuple[float, float, float]:
    """Calculate total EDP from CMEs"""
    adjusted_cmes = get_cmes_full_model_all_layers(cmes, LLAMA_3_8B, stage)
    
    total_latency = sum(cme.latency_total2 for cme in adjusted_cmes) / 1e9  # Latency is in cycles, assume 1GHz clock
    total_energy = sum(cme.energy_total for cme in adjusted_cmes) / 1e12  # Energy is in pJ
    total_edp = total_energy * total_latency    # EDP in Js
    
    return total_energy, total_latency, total_edp

def get_architecture_results(base_path: str, model_name: str, batch_size: int, 
                           prefill_tokens: int, decode_tokens: int, quant: str) -> Dict[str, Dict[str, float]]:
    """Get results for all architectures"""
    results = {}
    
    # List of architectures to compare
    architectures = [
        "generic_array_32b",
        "tpu_like",
        "tpu_like_onchip",
        "generic_array_32b_onchip"
    ]
    
    # For each architecture
    for arch in architectures:
        results[arch] = {}
        
        # For each stage (prefill and decode)
        for stage in Stage:
            # Construct the path to the pickle file
            if stage == Stage.PREFILL:
                pickle_path = os.path.join(
                    base_path,
                    f"{model_name}_B={batch_size}_tokens=({prefill_tokens}+{decode_tokens})_{quant}_{stage}_{arch}/cmes.pickle"
                )
            else:
                pickle_path = os.path.join(
                    base_path,
                    f"{model_name}_B={batch_size}_tokens=({prefill_tokens}+{decode_tokens})_{quant}_{stage}_{arch}/cmes.pickle"
                )
            
            # Load and process the CMEs
            try:
                with open(pickle_path, "rb") as fp:
                    cmes = pickle.load(fp)
                
                energy, latency, edp = get_total_edp(cmes, model_name, stage)
                results[arch][stage] = {
                    "energy": energy,
                    "latency": latency,
                    "edp": edp
                }
            except FileNotFoundError:
                print(f"Warning: Could not find file {pickle_path}")
                results[arch][stage] = None
    
    return results

def plot_edp_comparison(results: Dict[str, Dict[str, float]], model_name: str, 
                       batch_size: int, prefill_tokens: int, decode_tokens: int, quant: str):
    """Plot EDP comparison across architectures"""
    # Prepare data for plotting
    architectures = list(results.keys())
    stages = [stage.name for stage in Stage]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot EDP for each stage
    x = range(len(architectures))
    width = 0.35
    
    for i, stage in enumerate(stages):
        edp_values = [results[arch][Stage[stage]]["edp"] if results[arch][Stage[stage]] else 0 
                     for arch in architectures]
        bars = ax1.bar([xi + i*width for xi in x], edp_values, width, label=stage)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
    
    ax1.set_ylabel('Energy-Delay Product (J * s)')
    ax1.set_title('EDP Comparison Across Architectures')
    ax1.set_xticks([xi + width/2 for xi in x])
    ax1.set_xticklabels(architectures, rotation=45)
    ax1.legend()
    
    # Plot energy and latency breakdown
    for i, stage in enumerate(stages):
        energy_values = [results[arch][Stage[stage]]["energy"] if results[arch][Stage[stage]] else 0 
                        for arch in architectures]
        latency_values = [results[arch][Stage[stage]]["latency"] if results[arch][Stage[stage]] else 0 
                         for arch in architectures]
        
        energy_bars = ax2.bar([xi + i*width for xi in x], energy_values, width, label=f'{stage} Energy')
        latency_bars = ax2.bar([xi + i*width for xi in x], latency_values, width, bottom=energy_values, 
                label=f'{stage} Latency')
        
        # Add value labels on top of bars
        for bar in energy_bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        for bar in latency_bars:
            height = bar.get_height()
            bottom = bar.get_y()
            ax2.text(bar.get_x() + bar.get_width()/2., bottom + height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
    
    ax2.set_ylabel('Energy (J) / Latency (s)')
    ax2.set_title('Energy and Latency Breakdown')
    ax2.set_xticks([xi + width/2 for xi in x])
    ax2.set_xticklabels(architectures, rotation=45)
    ax2.legend()
    
    # Add overall title
    plt.suptitle(f'Architecture Comparison for {model_name}\n'
                f'B={batch_size}, tokens={prefill_tokens}+{decode_tokens}, {quant}')
    
    # Adjust layout and save
    plt.tight_layout()
    output_dir = "outputs/edp_comparison"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 
                            f"{model_name}_B={batch_size}_tokens={prefill_tokens}+{decode_tokens}_{quant}_edp.png"))
    plt.close()

def main():
    # Configuration
    base_path = "outputs/exp_compare_arch_vs_TPU_onchip_new"
    model_name = "Llama3-8B"
    batch_size = 1
    prefill_tokens = 256
    decode_tokens = 256
    quant = "W8A8"
    
    # Get results for all architectures
    results = get_architecture_results(
        base_path, model_name, batch_size, prefill_tokens, decode_tokens, quant
    )
    
    # Plot the comparison
    plot_edp_comparison(
        results, model_name, batch_size, prefill_tokens, decode_tokens, quant
    )

if __name__ == "__main__":
    main() 