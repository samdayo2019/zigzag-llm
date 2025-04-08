"""
Plot EDP comparison across different architectures for a given model configuration.
"""

import os
import sys
import pickle
import matplotlib.pyplot as plt
import subprocess
from typing import Dict, List, Tuple

sys.path.append(os.getcwd())
from src.config import LLAMA_3_8B, OPT_125M, LLMConfig
from src.util import CME_T, Stage

def get_cmes_full_model_all_layers(cmes: list[CME_T], model: LLMConfig, stage: Stage = Stage.PREFILL):
    """Generalize the zigzag results (for single layers) to a full LLM
    @param prefill: whether the results are from a prefill or decode phase simulation"""

    number_of_runs = 1 if stage == Stage.PREFILL else model.decode_size
    return [cme * model.get_post_simulation_multiplier(cme.layer.name) * number_of_runs for cme in cmes]

def get_total_edp(cmes: List[dict], model_name: str, stage: Stage) -> Tuple[float, float, float]:
    """Calculate total EDP from CMEs"""
    adjusted_cmes = get_cmes_full_model_all_layers(cmes, OPT_125M, stage)
    
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
        "simple_kv_accelerator",
        "simple_kv_accelerator low",
        "simple_kv_accelerator SRAM"
    ]
    
    leakage_current = [1e-18 * 20, 30e-12] # 0.02fA for gain cell and 30pA for SRAM (6T cell)
    leakage_ratio = 0.2 # 20% of leakage current from the memory cell itself
    # For each architecture
    for arch in architectures:
        if arch == "simple_kv_accelerator" or arch == "simple_kv_accelerator low":
            leakage = leakage_current[0]
        else: 
            leakage = leakage_current[1]
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
                    "edp": edp,
                    "leakage_current": leakage,
                    "leakage_ratio": leakage_ratio
                }
            except FileNotFoundError:
                print(f"Warning: Could not find file {pickle_path}")
                results[arch][stage] = None
    
    return results

def plot_edp_comparison(results: Dict[str, Dict[str, float]], model_name: str, 
                       batch_size: int, prefill_tokens: int, decode_tokens: int, quant: str):
    """Plot EDP, Energy, and Latency comparisons across architectures"""
    # Prepare data for plotting
    architectures = list(results.keys())
    stages = [stage.name for stage in Stage]
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # # Plot EDP for each stage
    x = range(len(architectures))
    width = 0.35
    
    # # First subplot: EDP
    # for i, stage in enumerate(stages):
    #     edp_values = [results[arch][Stage[stage]]["edp"] if results[arch][Stage[stage]] else 0 
    #                  for arch in architectures]
    #     bars = ax1.bar([xi + i*width for xi in x], edp_values, width, label=stage)
        
    #     # Add value labels on top of bars
    #     for bar in bars:
    #         height = bar.get_height()
    #         ax1.text(bar.get_x() + bar.get_width()/2., height,
    #                 f'{height:.5f}',
    #                 ha='center', va='bottom')
    
    # ax1.set_ylabel('Energy-Delay Product (J * s)')
    # ax1.set_title('EDP Comparison')
    # ax1.set_xticks([xi + width/2 for xi in x])
    # ax1.set_xticklabels(architectures, rotation=45)
    # ax1.legend()
    
    # Second subplot: Energy
    for i, stage in enumerate(stages):
        energy_values = [results[arch][Stage[stage]]["energy"] if results[arch][Stage[stage]] else 0 
                        for arch in architectures]
        bars = ax2.bar([xi + i*width for xi in x], energy_values, width, label=stage)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.5f}',
                    ha='center', va='bottom')
    
    ax2.set_ylabel('Energy (J)')
    ax2.set_title('Energy Comparison')
    ax2.set_xticks([xi + width/2 for xi in x])
    ax2.set_xticklabels(architectures, rotation=45)
    ax2.legend()
    
    # Third subplot: Latency
    for i, stage in enumerate(stages):
        latency_values = [results[arch][Stage[stage]]["latency"] if results[arch][Stage[stage]] else 0 
                         for arch in architectures]
        bars = ax3.bar([xi + i*width for xi in x], latency_values, width, label=stage)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.5f}',
                    ha='center', va='bottom')
    
    ax3.set_ylabel('Latency (s)')
    ax3.set_title('Latency Comparison')
    ax3.set_xticks([xi + width/2 for xi in x])
    ax3.set_xticklabels(architectures, rotation=45)
    ax3.legend()
    
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
    subprocess.run(["python", "exp_compare_arch.py"])

    # Configuration
    base_path = "outputs/athena_results"
    model_name = "OPT-125M"
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