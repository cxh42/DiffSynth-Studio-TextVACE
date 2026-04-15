import torch
import torch.utils.checkpoint as _cp

# Disable ALL checkpoint determinism checks for ZeRO-3 compatibility
# ZeRO-3 parameter gathering produces tensors with different strides during recomputation
_cp._DEFAULT_DETERMINISM_MODE = 'none'
_cp.set_checkpoint_early_stop(False)


def create_custom_forward(module):
    def custom_forward(*inputs, **kwargs):
        return module(*inputs, **kwargs)
    return custom_forward


def gradient_checkpoint_forward(
    model,
    use_gradient_checkpointing,
    use_gradient_checkpointing_offload,
    *args,
    **kwargs,
):
    if use_gradient_checkpointing_offload:
        with torch.autograd.graph.save_on_cpu():
            model_output = _cp.checkpoint(
                create_custom_forward(model),
                *args,
                **kwargs,
                use_reentrant=False,
                determinism_check='none',
            )
    elif use_gradient_checkpointing:
        model_output = _cp.checkpoint(
            create_custom_forward(model),
            *args,
            **kwargs,
            use_reentrant=False,
            determinism_check='none',
        )
    else:
        model_output = model(*args, **kwargs)
    return model_output
