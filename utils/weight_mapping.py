import torch
import torch.nn as nn
import logging

def load_pretrained_weights(quant_model: nn.Module, float_model: nn.Module):
    """
    Maps weights from a floating-point model to a quantized Brevitas model.
    
    Handles automatic reshaping of depthwise conv weights that may be 
    flattened to 1D in the source checkpoint.
    """
    logging.info("Mapping pretrained floating-point weights to quantized model...")
    float_state_dict = float_model.state_dict()
    quant_state_dict = quant_model.state_dict()

    filtered_dict = {}
    for k, v in float_state_dict.items():
        if k in quant_state_dict:
            target_shape = quant_state_dict[k].shape
            # Auto-reshape flattened depthwise conv weights (1D -> 4D)
            if v.dim() == 1 and len(target_shape) == 4:
                v = v.reshape(target_shape)
            filtered_dict[k] = v

    missing_keys, unexpected_keys = quant_model.load_state_dict(filtered_dict, strict=False)
    
    logging.info(f"Successfully mapped {len(filtered_dict)} tensors.")
    if missing_keys:
        logging.debug(f"Missing keys (expected for Brevitas): {missing_keys}")
    if unexpected_keys:
        logging.warning(f"Unexpected keys: {unexpected_keys}")

    return quant_model
