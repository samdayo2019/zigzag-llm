from copy import deepcopy

BATCH_SIZE = 8


class LLMConfig:
    def __init__(
        self,
        seq_len: int,
        embedding_dim: int,
        dim_ff: int,
        num_query_heads: int,
        num_layer: int,
        group_size: int = 1,
        batch_size: int = 1,
        vocab_size: int = 1000,
        name: str = "",
    ):
        self.batch_size = batch_size
        self.__seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.dim_ff = dim_ff
        self.num_query_heads = num_query_heads # number of query heads
        self.num_kv_heads = num_query_heads // group_size # number of key-value heads
        self.num_layer = num_layer
        self.head_size = embedding_dim // num_query_heads
        self.vocab_size = vocab_size
        self.group_size = group_size
        self.__name = name

        # Simulate prefill with half of the context window.
        self.prefill_size = self.__seq_len // 2
        self.decode_size = self.__seq_len // 2
        # print("NEW LLMConfig")

    @property
    def name(self):
        return f"{self.__name}"

    @property
    def seq_len(self):
        return self.__seq_len

    @property
    def decode_idx(self):
        """To simulate the model in decode phase, only a single run (for a single) token is executed"""
        return self.prefill_size + self.decode_size // 2

    @property
    def parameterized_name(self):
        return f"{self.name.replace('.', '_')}_B={self.batch_size}_tokens=({self.prefill_size}+{self.decode_size})"

    def to_simulatable_config(self):
        """Return a new LLMConfig instance with reduced parameters to make the simulation go faster. The results
        can then be multiplied to get the actual energy/latency values"""
        cfg = deepcopy(self)
        # cfg.num_layer = 1
        cfg.num_layer = 1
        # cfg.num_query_heads = 1  # Keep the original `head_size`!
        # cfg.num_query_heads = cfg.num_query_heads // 2
        # cfg.num_kv_heads = 1
        # cfg.num_kv_heads = cfg.num_kv_heads // 2
        return cfg

    def get_post_simulation_multiplier(self, layer: str):
        """The model is simulated with reduced parameters i.e. only one layer. This function returns the factor with
        which the results for the given layer have to be multiplied in order to come to the result for the full model
        Moreover, the results are normalized to a single inference instead of a full batch"""
        assert self.num_layer > 1, "Is this method called on a `simulatable` config?"
        # K, Q, V and output projection
        if "_proj" in layer:
            return self.num_layer / self.batch_size
        elif "mul_" in layer:
            return self.num_layer / self.batch_size
        # Special case: gate layer in Llama models
        elif "feedforward_expand" in layer and "llama" in self.name.lower():
            return 2 * self.num_layer / self.batch_size
        elif "feedforward_" in layer:
            return self.num_layer / self.batch_size
        else:
            return 1


class QuantConfig:
    def __init__(self, weight_bits: int, act_bits: int, output_bits: int | None = None):
        self.weight_bits = weight_bits
        self.act_bits = act_bits
        self.intermediate_output_bits = output_bits if output_bits is not None else 2 * act_bits

    @property
    def name(self):
        return f"W{self.weight_bits}A{self.act_bits}"


W1A8 = QuantConfig(1, 8, 16)
W4A8 = QuantConfig(4, 8, 16)
W8A8 = QuantConfig(8, 8, 16)
W4A16 = QuantConfig(4, 16, 16)
W1A32 = QuantConfig(1, 32, 32)
W16A32 = QuantConfig(16, 32, 32)
W32A32 = QuantConfig(32, 32, 32)
W16A16 = QuantConfig(16, 16, 16)

LLAMA_1_7B = LLMConfig(
    batch_size=BATCH_SIZE,
    seq_len=2048,
    embedding_dim=4096,
    dim_ff=11_008,
    num_query_heads=32,
    num_layer=32,
    vocab_size=32_000,
    name="Llama1-7B",
)

LLAMA_1_13B = LLMConfig(
    batch_size=BATCH_SIZE,
    seq_len=2048,
    embedding_dim=5120,
    dim_ff=13_824,
    num_query_heads=40,
    num_layer=40,
    vocab_size=32_000,
    name="Llama1-13B",
)

LLAMA_1_30B = LLMConfig(
    batch_size=BATCH_SIZE,
    seq_len=2048,
    embedding_dim=6_656,
    dim_ff=17_920,
    num_query_heads=52,
    num_layer=60,
    vocab_size=32_000,
    name="Llama1-30B",
)

LLAMA_2_7B = LLMConfig(
    batch_size=BATCH_SIZE,
    seq_len=4096,
    embedding_dim=4096,
    dim_ff=11_008,
    num_query_heads=32,
    num_layer=32,
    group_size=1,
    vocab_size=32_000,
    name="Llama2-7B",
)

LLAMA_2_13B = LLMConfig(
    batch_size=BATCH_SIZE,
    seq_len=4096,
    embedding_dim=5120,
    dim_ff=13_824,
    num_query_heads=40,
    num_layer=40,
    vocab_size=32_000,
    name="Llama2-13B",
)

LLAMA_3_8B = LLMConfig(
    batch_size=BATCH_SIZE,
    seq_len=4096,
    embedding_dim=4096,
    dim_ff=14336,
    num_query_heads=32,
    group_size=4,
    num_layer=32,
    vocab_size=128_000,
    name="Llama3-8B",
)

LLAMA_3_70B = LLMConfig(
    batch_size=BATCH_SIZE,
    seq_len=4096,
    embedding_dim=8192,
    dim_ff=28672,
    num_query_heads=64,
    group_size=8,
    num_layer=80,
    vocab_size=128_000,
    name="Llama3-70B",
)

LLAMA_3_405B = LLMConfig(
    batch_size=BATCH_SIZE,
    seq_len=4096,
    embedding_dim=16384,
    dim_ff=53248,
    num_query_heads=128,
    group_size=16,
    num_layer=126,
    vocab_size=128_000,
    name="Llama3-405B",
)

LLAMA_3_3T = LLMConfig(
    batch_size=BATCH_SIZE,
    seq_len=4096,
    embedding_dim=32768,
    dim_ff=114688,
    num_query_heads=256,
    group_size=32,
    num_layer=256,
    vocab_size=128_000,
    name="Llama3-3T",
)

OPT_125M = LLMConfig(
    batch_size=BATCH_SIZE,
    seq_len=2048,
    embedding_dim=768,
    dim_ff=3072,
    num_query_heads=12,
    num_layer=12,
    vocab_size=50_272,
    name="OPT-125M",
)


OPT_1_3B = LLMConfig(
    batch_size=BATCH_SIZE,
    seq_len=2048,
    embedding_dim=2048,
    dim_ff=8_192,
    num_query_heads=32,
    num_layer=24,
    vocab_size=50_272,
    name="OPT-1.3B",
)

OPT_2_7B = LLMConfig(
    batch_size=BATCH_SIZE,
    seq_len=2048,
    embedding_dim=2560,
    dim_ff=10240,
    num_query_heads=32,
    num_layer=32,
    vocab_size=50_272,
    name="OPT-2.7B",
)

OPT_6_7B = LLMConfig(
    batch_size=BATCH_SIZE,
    seq_len=2048,
    embedding_dim=4096,
    dim_ff=16384,
    num_query_heads=32,
    num_layer=32,
    vocab_size=50_272,
    name="OPT-6.7B",
)

OPT_13B = LLMConfig(
    batch_size=BATCH_SIZE,
    seq_len=2048,
    embedding_dim=5120,
    dim_ff=20480,
    num_query_heads=40,
    num_layer=40,
    vocab_size=50_272,
    name="OPT-13B",
)

OPT_30B = LLMConfig(
    batch_size=BATCH_SIZE,
    seq_len=2048,
    embedding_dim=7_168,
    dim_ff=28_672,
    num_query_heads=56,
    num_layer=48,
    vocab_size=50_272,
    name="OPT-30B",
)

GPT3_175B = LLMConfig(
    batch_size=BATCH_SIZE,
    seq_len=2048,
    embedding_dim=12_288,
    dim_ff=49_152,
    num_query_heads=96,
    num_layer=96,
    vocab_size=50257,
    name="GPT3-175B",
)

ALL_MODELS = [
    # LLAMA_1_7B,
    # LLAMA_1_13B,
    # LLAMA_1_30B,
    LLAMA_2_7B,
    # LLAMA_2_13B,
    OPT_125M,
    # OPT_1_3B,
    # OPT_2_7B,
    # OPT_6_7B,
    # OPT_13B,
    # OPT_30B,
    GPT3_175B,
    LLAMA_3_8B,
    LLAMA_3_70B,
    LLAMA_3_405B,
    LLAMA_3_3T,
]

PAPER_MODELS = [
    LLAMA_2_7B,
    OPT_125M,
    GPT3_175B,
    LLAMA_3_405B,
]
