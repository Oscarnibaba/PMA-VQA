import os
import torch
import logging
from collections import OrderedDict


def load_checkpoint(model, filename, strict=False, logger=None):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The checkpoint loaded.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if not os.path.isfile(filename):
        logger.error(f"Checkpoint file {filename} does not exist!")
        raise FileNotFoundError(f"Checkpoint file {filename} does not exist!")

    logger.info(f"Loading checkpoint from {filename}")
    checkpoint = torch.load(filename, map_location='cpu')

    # get state_dict from checkpoint
    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        logger.error(f"No state_dict found in checkpoint file {filename}")
        raise ValueError(f"No state_dict found in checkpoint file {filename}")

    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    # load state_dict
    load_state_dict(model, state_dict, strict, logger)
    logger.info(f"Checkpoint loaded successfully from {filename}")

    return checkpoint


def load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.

    Args:
        module (Module): Module to receive the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function.
        logger (:mod:`logging.Logger` or None): The logger for error message.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys, err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None  # break load->load reference cycle

    # ignore "num_batches_tracked" of BN layers
    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    if strict:
        if len(err_msg) > 0:
            err_msg.insert(
                0, 'The model and loaded state dict do not match exactly\n')
            err_msg = '\n'.join(err_msg)
            raise RuntimeError(err_msg)
    elif len(err_msg) > 0:
        logger.warning(err_msg)
