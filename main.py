"""
Make a single plot for 1 model on 1 architecture
"""

import os
import sys

sys.path.append(os.getcwd())
from src.config import OPT_125M, W32A32, W16A16, W8A8, LLAMA_3_8B, LLAMA_3_70B, LLAMA_3_405B, LLAMA_3_3T, LLAMA_2_7B
from src.export_onnx import Stage
from src.plots import plot_energy_and_latency
from src.simulation import run_simulation
from src.util import (
    CME_T,
    get_cmes_full_model_from_pickle,
    get_experiment_id,
)

model = LLAMA_3_8B
model.batch_size = 32
model.prefill_size = 1024
model.decode_size = 1024
quant = W8A8
accelerator = "tpu_8b_hbm"
mapping_path = "inputs/mapping/weight_unrolled_256.yaml"
out_path = "outputs/main"


def run_experiment():
    for stage in Stage:
        run_simulation(
            model=model,
            stage=stage,
            quant=quant,
            accelerator_name=accelerator,
            mapping_path=mapping_path,
            output_dir=out_path,
        )


if __name__ == "__main__":
    print("Running experiment...")
    run_experiment()

    cmes_per_group: list[list[CME_T]] = []

    for stage in Stage:
        experiment_name = get_experiment_id(model, stage, quant, accelerator)
        pickle_filename = f"{out_path}/{experiment_name}/cmes.pickle"
        cmes_full_model = get_cmes_full_model_from_pickle(pickle_filename, model, stage)
        cmes_per_group.append(cmes_full_model)

    plot_energy_and_latency(
        cmes_per_group,
        supergroups=["Prefill", "Decode"],
        title=f"{model.name} ({quant.name})",
        filename=f"{out_path}/energy_and_latency_{model.name}.png",
    )
