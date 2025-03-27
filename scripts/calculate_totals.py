"""
Calculate total latency and energy from CMEs.
"""

import os
import sys
import pickle
from typing import Tuple, Dict, List

sys.path.append(os.getcwd())
from src.plot_util import PlotCMEDetailed
from src.util import CME_T, Stage, get_cmes_full_model_from_pickle
from src.config import *

def get_cmes_full_model_all_layers(cmes: list[CME_T], model: LLMConfig, stage: Stage = Stage.PREFILL):
    """Generalize the zigzag results (for single layers) to a full LLM
    @param prefill: whether the results are from a prefill or decode phase simulation"""

    number_of_runs = 1 if stage == Stage.PREFILL else model.decode_size
    return [cme * model.get_post_simulation_multiplier(cme.layer.name) * number_of_runs for cme in cmes]

def get_total_latency(cme: CME_T) -> float:
    """Calculate total latency from a CME"""
    return cme.latency_total2  # This includes all latency components

def get_total_energy(cme: CME_T) -> float:
    """Calculate total energy from a CME"""
    data = cme.__jsonrepr__()["outputs"]["energy"]
    total_energy = data["operational_energy"]  # MAC energy
    
    # Determine if this is a weight layer
    is_weight_layer = "key_proj" in cme.layer.name or "value_proj" in cme.layer.name
    
    # Add memory energy based on layer type
    if is_weight_layer:
        # For weight layers (e.g., key_proj, value_proj)
        # Weight read energy
        total_energy += PlotCMEDetailed.get_mem_energy(data, "W", 0, is_read=True)  # RF
        total_energy += PlotCMEDetailed.get_mem_energy(data, "W", 1, is_read=True)  # SRAM
        total_energy += PlotCMEDetailed.get_mem_energy(data, "W", 2, is_read=True)  # DRAM
        
        # Input read energy
        total_energy += PlotCMEDetailed.get_mem_energy(data, "I", 0, is_read=True)  # RF
        total_energy += PlotCMEDetailed.get_mem_energy(data, "I", 1, is_read=True)  # SRAM
        total_energy += PlotCMEDetailed.get_mem_energy(data, "I", 2, is_read=True)  # DRAM
        
        # Output write energy
        total_energy += PlotCMEDetailed.get_mem_energy(data, "O", 0, is_read=False)  # RF
        total_energy += PlotCMEDetailed.get_mem_energy(data, "O", 1, is_read=False)  # SRAM
        total_energy += PlotCMEDetailed.get_mem_energy(data, "O", 2, is_read=False)  # DRAM
    else:
        # For non-weight layers (e.g., mul_qk_t, mul_logits_v)
        # Input read energy
        total_energy += PlotCMEDetailed.get_mem_energy(data, "I", 0, is_read=True)  # RF
        total_energy += PlotCMEDetailed.get_mem_energy(data, "I", 1, is_read=True)  # SRAM
        total_energy += PlotCMEDetailed.get_mem_energy(data, "I", 2, is_read=True)  # DRAM
        
        # Weight read energy (in act2)
        total_energy += PlotCMEDetailed.get_mem_energy(data, "W", 0, is_read=True)  # RF
        total_energy += PlotCMEDetailed.get_mem_energy(data, "W", 1, is_read=True)  # SRAM
        total_energy += PlotCMEDetailed.get_mem_energy(data, "W", 2, is_read=True)  # DRAM
        
        # Output write energy
        total_energy += PlotCMEDetailed.get_mem_energy(data, "O", 0, is_read=False)  # RF
        total_energy += PlotCMEDetailed.get_mem_energy(data, "O", 1, is_read=False)  # SRAM
        total_energy += PlotCMEDetailed.get_mem_energy(data, "O", 2, is_read=False)  # DRAM
    
    return total_energy

def organize_cmes_by_layer(cmes: List[CME_T]) -> Dict[str, List[CME_T]]:
    """Organize CMEs by their layer names"""
    organized: Dict[str, List[CME_T]] = {}
    for cme in cmes:
        layer_name = cme.layer.name
        if layer_name not in organized:
            organized[layer_name] = []
        organized[layer_name].append(cme)
    return organized

def print_layer_breakdown(organized_cmes: Dict[str, List[CME_T]]):
    """Print a breakdown of CMEs by layer"""
    print("\nLayer Breakdown:")
    print("-" * 80)
    for layer_name, layer_cmes in organized_cmes.items():
        print(f"\nLayer: {layer_name}")
        print(f"Number of CMEs: {len(layer_cmes)}")
        for i, cme in enumerate(layer_cmes):
            latency = get_total_latency(cme)
            energy = get_total_energy(cme)
            print(f"  CME {i+1}:")
            print(f"    Latency: {latency:.2f} cycles")
            print(f"    Energy: {energy:.2f} pJ")
            print(f"    Memory Size: {cme.layer.memory_size if hasattr(cme.layer, 'memory_size') else 'N/A'}")
            print(f"    Memory Bandwidth: {cme.layer.memory_bandwidth if hasattr(cme.layer, 'memory_bandwidth') else 'N/A'}")
            print()

def calculate_totals(pickle_file: str, model: LLMConfig, stage: Stage) -> Tuple[float, float]:
    """Calculate total latency and energy from a pickle file containing CMEs"""
    with open(pickle_file, "rb") as fp:
        cmes = pickle.load(fp)
    
    adjusted_cmes = get_cmes_full_model_all_layers(cmes, model, stage)

    print(f"Total Number of CMEs: {len(adjusted_cmes)}")
    
    # Organize and print CMEs by layer
    organized_cmes = organize_cmes_by_layer(adjusted_cmes)
    print_layer_breakdown(organized_cmes)
    
    total_latency = sum(get_total_latency(cme) for cme in adjusted_cmes)
    total_energy = sum(get_total_energy(cme) for cme in adjusted_cmes)
    
    print("\nOverall Totals:")
    print("-" * 80)
    print(f"Total Latency: {total_latency:.2f} cycles")
    print(f"Total Energy: {total_energy:.2f} pJ")
    
    return total_latency, total_energy

if __name__ == "__main__":
    # Example usage
    pickle_file = "outputs/exp_compare_arch_vs_TPU_onchip_new/Llama3-8B_B=8_tokens=(256+256)_W4A8_prefill_tpu_like/cmes.pickle"  # Replace with your pickle file path
    # parse pickle file name to get model name
    model = LLAMA_3_8B
    stage = Stage.PREFILL
    total_latency, total_energy = calculate_totals(pickle_file, model, stage) 