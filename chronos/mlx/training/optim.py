"""AdamW with FP32 accumulation/state and low-precision forward weights."""
import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_map_with_path


class MixedPrecisionAdamW(optim.Adam):
    """Match the Torch AdamW recipe without rounding away small BF16 updates.

    The FP32 master copy is required only for BF16/FP16 parameters. Its memory
    cost is reported separately by the UI estimator. Optimizer snapshots include
    masters and moments, so restoring a snapshot restores the complete update.
    """

    def __init__(self, learning_rate, weight_decay=.01, betas=(.9, .95), eps=1e-8):
        super().__init__(learning_rate, betas=betas, eps=eps, bias_correction=True)
        self.weight_decay = float(weight_decay)

    def init_single(self, parameter, state):
        state["m"] = mx.zeros(parameter.shape, dtype=mx.float32)
        state["v"] = mx.zeros(parameter.shape, dtype=mx.float32)
        if parameter.dtype in (mx.bfloat16, mx.float16):
            state["master"] = parameter.astype(mx.float32)

    def init(self, parameters):
        super().init(parameters)
        def assign_decay(path, parameter, state):
            # Same exclusion as the Torch recipe: embeddings, norms and biases.
            state["decay"] = parameter.ndim >= 2 and not path.endswith("embed_tokens.weight")
        tree_map_with_path(assign_decay, parameters, self.state)

    def apply_gradients(self, gradients, parameters):
        # MLX's default initializes from gradients; masters must start at WEIGHTS.
        if not self._initialized:
            self.init(parameters.trainable_parameters() if hasattr(parameters, "trainable_parameters") else parameters)
        return super().apply_gradients(gradients, parameters)

    def apply_single(self, gradient, parameter, state):
        master = state.get("master", parameter.astype(mx.float32))
        if state["decay"]:
            master = master * (1 - self.learning_rate * self.weight_decay)
        updated = super().apply_single(gradient.astype(mx.float32), master, state)
        if "master" in state:
            state["master"] = updated
        return updated.astype(parameter.dtype)
