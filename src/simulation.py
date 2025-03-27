import os
import pickle

from zigzag import api
from zigzag.visualization.results.plot_cme import (
    bar_plot_cost_model_evaluations_breakdown,
)

from src.config import LLMConfig, QuantConfig
from src.export_onnx import Stage, export_transformer_to_onnx
from src.plots import (
    plot_energy_clean,
    plot_latency_clean,
)
from src.util import (
    generalize_layer_name,
    get_accelerator_path,
    get_cmes_full_model,
    get_cmes_to_plot,
    get_experiment_id,
    get_onnx_path,
)

from copy import deepcopy

def run_simulation(
    model: LLMConfig,
    stage: Stage,
    quant: QuantConfig,
    accelerator_name: str,
    mapping_path: str,
    output_dir: str,
    *,
    opt_criterion: str = "EDP",
    experiment_id: str | None = None,
    onnx_path: str | None = None,
    dump_path: str | None = None,
    pickle_filename: str | None = None,
):
    assert model.num_layer >= 1, "Is this a `simulatable` config?"
    model_for_simulation = model.to_simulatable_config() #Get new config with reduced parameters (full query heads, full key-value heads, 1 layer)
    # model_for_simulation = deepcopy(model)

    if experiment_id is None:
        # extract experiment id from model, stage, quant, accelerator_name
        experiment_id = get_experiment_id(model, stage, quant, accelerator_name)

    if onnx_path is None:
        # extract onnx save path from model, stage, quant
        onnx_path = get_onnx_path(model_for_simulation, stage, quant)

    if dump_path is None:
        # extract dump path from output_dir, experiment_id
        dump_path = f"{output_dir}/{experiment_id}"

    if pickle_filename is None:
        # Pickle file name default None, so we generate dump path
        pickle_filename = f"{dump_path}/cmes.pickle" # which layers to include in the cost model

    # print(f"--- Testing New Simulation. Here is KV Heads {model.num_kv_heads} ---")
    # print(f"--- Running {experiment_id} ---")
    # print(f"--- Pickle file: {pickle_filename} ---")
    
    print(f"--- Model layers: {model_for_simulation.num_layer} ---")
    print(f"--- Model query heads: {model_for_simulation.num_query_heads} ---")

    if not os.path.exists(onnx_path):
        # if the onnx file does not exist, create it
        export_transformer_to_onnx(model_for_simulation, quant, path=onnx_path, stage=stage)

    # use the Zigzag API to get the hardware performance
    api.get_hardware_performance_zigzag(
        workload=onnx_path,
        accelerator=get_accelerator_path(accelerator_name),
        mapping=mapping_path,
        opt=opt_criterion,
        dump_folder=dump_path,
        pickle_filename=pickle_filename,
        nb_spatial_mappings_generated=10, # get 10 spatial mappings
    )

    with open(pickle_filename, "rb") as fp:
        cmes = pickle.load(fp)

    # Plots for single layers
    # print(f" Original CMEs: {cmes}")
    cmes_to_plot = get_cmes_to_plot(cmes)
    bar_plot_cost_model_evaluations_breakdown(cmes, save_path=f"{dump_path}/all_layers_single.png")
    bar_plot_cost_model_evaluations_breakdown(cmes_to_plot, save_path=f"{dump_path}/interesting_layers_single.png")

    # Compute generalized results for full LLM
    complete_result_cmes = get_cmes_full_model(cmes_to_plot, model, stage)
    bar_plot_cost_model_evaluations_breakdown(
        complete_result_cmes, save_path=f"{dump_path}/interesting_layers_full.png"
    )

    plot_energy_clean(complete_result_cmes, f"{dump_path}/energy.png")
    plot_latency_clean(complete_result_cmes, f"{dump_path}/latency.png")

    # Save which layers are plotted
    with open(f"{dump_path}/info.txt", "w") as f:
        f.write("Layers shown in plot interesting_layers_single:\n")
        for idx, cme in enumerate(cmes_to_plot):
            f.write(f"\t{idx}: {cme.layer.name}\n")
        f.write(
            "\tNote: the linear projection shows a single projection (e.g. key) for ALL heads. The MatMuls "
            "(attention and logits) are shown for a SINGLE head.\n"
        )
        f.write("Components shown in plot interesting_layers_full:\n")
        for idx, cme in enumerate(cmes_to_plot):
            f.write(f"\t{idx}: {generalize_layer_name(cme.layer.name)}\n")
        f.write("Components shown in plot all_layers_single:\n")
        for idx, cme in enumerate(cmes):
            f.write(f"\t{idx}: {cme.layer.name}\n")
