import os
import numpy as np
from timm import optim
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import torch
from torch.utils.tensorboard import SummaryWriter
import logging as log
from box import Box
import shutil
from pathlib import Path
import torch.nn.functional as F
import datetime
import random
from accelerate import Accelerator
from sklearn.metrics import precision_score, recall_score


def focal_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2,
        reduction: str = "none", ) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.
    """
    # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
    # p = torch.sigmoid(inputs) #My pred is already from 0 to 1 no need to use binary_cross_entropy_with_sigmoid
    p = inputs
    ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss


def truncate_seq(seq, max_len):  # padding
    return [seqi[:max_len] for seqi in seq]


def f1_max(pred, target):
    """
    copied from https://torchdrug.ai/docs/_modules/torchdrug/metrics/metric.html#f1_max
    F1 score with the optimal threshold.

    This function first enumerates all possible thresholds for deciding positive and negative
    samples, and then pick the threshold with the maximal F1 score.

    Parameters:
        pred (Tensor): predictions of shape :math:`(B, N)`
        target (Tensor): binary targets of shape :math:`(B, N)`
    """

    order = pred.argsort(descending=True, dim=1)
    target = target.gather(1, order)
    precision = target.cumsum(1) / torch.ones_like(target).cumsum(1)
    recall = target.cumsum(1) / (target.sum(1, keepdim=True) + 1e-10)
    is_start = torch.zeros_like(target).bool()
    is_start[:, 0] = 1
    is_start = torch.scatter(is_start, 1, order, is_start)

    all_order = pred.flatten().argsort(descending=True)
    order = order + torch.arange(order.shape[0], device=order.device).unsqueeze(1) * order.shape[1]
    order = order.flatten()
    inv_order = torch.zeros_like(order)
    inv_order[order] = torch.arange(order.shape[0], device=order.device)
    is_start = is_start.flatten()[all_order]
    all_order = inv_order[all_order]
    precision = precision.flatten()
    recall = recall.flatten()
    all_precision = precision[all_order] - \
                    torch.where(is_start, torch.zeros_like(precision), precision[all_order - 1])
    all_precision = all_precision.cumsum(0) / is_start.cumsum(0)
    all_recall = recall[all_order] - \
                 torch.where(is_start, torch.zeros_like(recall), recall[all_order - 1])
    all_recall = all_recall.cumsum(0) / pred.shape[0]
    all_f1 = 2 * all_precision * all_recall / (all_precision + all_recall + 1e-10)
    return all_f1.max()


def Fmax_func(predictions, targets, binnum=10):
    custom_thresholds = np.linspace(0, 1, binnum)[1:-1]
    precision_at_thresholds = []
    recall_at_thresholds = []
    num_samples = len(predictions)
    for sample_index in range(num_samples):  # Loop through each class
        preds = predictions[sample_index, :]

        # Calculate precision and recall at custom thresholds
        precisions = [precision_score(targets[sample_index, :], (preds >= threshold), zero_division=0) for threshold in
                      custom_thresholds]
        recalls = [recall_score(targets[sample_index, :], (preds >= threshold), zero_division=0) for threshold in
                   custom_thresholds]

        precision_at_thresholds.append(precisions)
        recall_at_thresholds.append(recalls)

    # Calculate average precision and recall
    avg_precision = np.mean(precision_at_thresholds, 0)
    avg_recall = np.mean(recall_at_thresholds, 0)
    f_scores = [2 * (p * r) / (p + r + 1e-8) for (p, r) in zip(avg_precision, avg_recall)]
    # Calculate F1-score using average precision and recall
    return np.max(f_scores)


def print_gpu_memory_allocation(logging):
    """
    Prints the memory allocation and caching for each available GPU using PyTorch.
    """
    if not torch.cuda.is_available():
        print("No GPUs detected!")
        return

    num_gpus = torch.cuda.device_count()

    for i in range(num_gpus):
        total_mem = torch.cuda.get_device_properties(i).total_memory / 1e9  # in GB
        allocated_mem = torch.cuda.memory_allocated(i) / 1e9  # in GB
        cached_mem = torch.cuda.memory_reserved(i) / 1e9  # in GB

        logging.info(f"GPU {i}:")
        logging.info(f"\tTotal Memory: {total_mem:.2f} GB")
        logging.info(f"\tAllocated Memory: {allocated_mem:.2f} GB")
        logging.info(f"\tCached (Reserved) Memory: {cached_mem:.2f} GB")
        logging.info("-" * 50)


def calculate_class_weights(class_samples):
    total_samples = sum(class_samples.values())
    class_weights = {}

    # Calculate weights using the inverse of the class frequencies
    for class_name, samples in class_samples.items():
        class_weights[class_name] = total_samples / samples

    # Normalize the weights so that the largest class has a weight of 1
    min_weight = min(class_weights.values())
    for class_name, weight in class_weights.items():
        class_weights[class_name] = weight / min_weight

    return class_weights


def calculate_class_weights_normalized(samples_dict):
    """
    Calculate the weights for each class based on the number of samples.

    :param samples_dict: Dictionary containing classes as keys and number of samples as values.
    :return: Dictionary containing classes as keys and their respective weights as values.
    """
    min_samples = min(samples_dict.values())

    weights_dict = {}
    for key, value in samples_dict.items():
        weights_dict[key] = min_samples / value

    return weights_dict


def load_configs(config, args=None):
    """
        Load the configuration file and convert the necessary values to floats.

        Args:
            config (dict): The configuration dictionary.

        Returns:
            The updated configuration dictionary with float values.
        """

    # Convert the dictionary to a Box object for easier access to the values.
    tree_config = Box(config)

    # Convert the necessary values to floats.
    tree_config.optimizer.lr = float(tree_config.optimizer.lr)
    tree_config.optimizer.decay.min_lr = float(tree_config.optimizer.decay.min_lr)
    tree_config.optimizer.weight_decay = float(tree_config.optimizer.weight_decay)
    tree_config.optimizer.eps = float(tree_config.optimizer.eps)
    # overwrite parameters if set through commandline
    #print("num_end_adapter_layers!!!!!!!!")
    #print(tree_config.encoder.adapter_h.num_end_adapter_layers)
    if args is not None:
        if args.result_path:
            tree_config.result_path = args.result_path

        if args.resume_path:
            tree_config.resume.resume_path = args.resume_path
            #tree_config.resume.enable = True #if set by args, the resume enable will be overwrite as True
        
        if args.num_end_adapter_layers:
            if not isinstance(args.num_end_adapter_layers, list):
                if "-" in args.num_end_adapter_layers:
                   args.num_end_adapter_layers = args.num_end_adapter_layers.split("-")
                else:
                   args.num_end_adapter_layers = [args.num_end_adapter_layers]
            
            args.num_end_adapter_layers = [int(x) for x in args.num_end_adapter_layers]
            tree_config.encoder.adapter_h.num_end_adapter_layers = args.num_end_adapter_layers

        if args.module_type:
            tree_config.encoder.adapter_h.module_type = args.module_type

    print("num_end_adapter_layers!!!!!!!!")
    print(tree_config.encoder.adapter_h.num_end_adapter_layers)

    print("freeze_adapter_layers!!!!!!!!")
    print(tree_config.encoder.adapter_h.freeze_adapter_layers)
    return tree_config


def prepare_saving_dir(configs, config_file_path):
    """
    Prepare a directory for saving a training results.

    Args:
        configs: A python box object containing the configuration options.

    Returns:
        str: The path to the directory where the results will be saved.
    """
    # Create a unique identifier for the run based on the current time.
    run_id = datetime.datetime.now().strftime('%Y-%m-%d__%H-%M-%S')

    # Add '_evaluation' to the run_id if the 'evaluate' flag is True.
    # if configs.evaluate:
    #     run_id += '_evaluation'

    # Create the result directory and the checkpoint subdirectory.
    result_path = os.path.abspath(os.path.join(configs.result_path, run_id))
    checkpoint_path = os.path.join(result_path, 'checkpoints')
    Path(result_path).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    # Copy the config file to the result directory.
    shutil.copy(config_file_path, result_path)

    # Return the path to the result directory.
    return result_path, checkpoint_path


def prepare_optimizer(net, configs, num_train_samples, logging):
    optimizer, scheduler = load_opt(net, configs, logging)
    if scheduler is None:
        if configs.optimizer.decay.first_cycle_steps:
            first_cycle_steps = configs.optimizer.decay.first_cycle_steps
        else:
            whole_steps = np.ceil(
                num_train_samples / configs.train_settings.grad_accumulation
            ) * configs.train_settings.num_epochs
            first_cycle_steps = np.ceil(whole_steps / configs.optimizer.decay.num_restarts)

        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=first_cycle_steps,
            cycle_mult=1.0,
            max_lr=configs.optimizer.lr,
            min_lr=configs.optimizer.decay.min_lr,
            warmup_steps=configs.optimizer.decay.warmup,
            gamma=configs.optimizer.decay.gamma)

    return optimizer, scheduler


def load_opt(model, config, logging):
    scheduler = None
    if config.optimizer.name.lower() == 'adabelief':
        opt = optim.AdaBelief(model.parameters(), lr=config.optimizer.lr, eps=config.optimizer.eps,
                              decoupled_decay=True,
                              weight_decay=config.optimizer.weight_decay, rectify=False)
    elif config.optimizer.name.lower() == 'adam':
        # opt = eval('torch.optim.' + config.optimizer.name)(model.parameters(), lr=config.optimizer.lr, eps=eps,
        #                                       weight_decay=config.optimizer.weight_decay)
        if config.optimizer.use_8bit_adam:
            import bitsandbytes
            logging.info('use 8-bit adamw')
            opt = bitsandbytes.optim.AdamW8bit(
                model.parameters(), lr=float(config.optimizer.lr),
                betas=(config.optimizer.beta_1, config.optimizer.beta_2),
                weight_decay=float(config.optimizer.weight_decay),
                eps=float(config.optimizer.eps),
            )
        else:
            opt = torch.optim.AdamW(
                model.parameters(), lr=float(config.optimizer.lr),
                betas=(config.optimizer.beta_1, config.optimizer.beta_2),
                weight_decay=float(config.optimizer.weight_decay),
                eps=float(config.optimizer.eps)
            )

    else:
        raise ValueError('wrong optimizer')
    return opt, scheduler


def load_checkpoints_only(checkpoint_path,  model):
    model_checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict1' in model_checkpoint:
        #to load old checkpoints that saved adapter_layer_dict as adapter_layer. 
        from collections import OrderedDict
        if np.sum(["adapter_layer_dict" in key for key in model_checkpoint['state_dict1'].keys()])==0: #using old checkpoints, need to rename the adapter_layer into adapter_layer_dict.adapter_0
             new_ordered_dict = OrderedDict()
             for key, value in model_checkpoint['state_dict1'].items():
                 if "adapter_layer_dict" not in key:
                   new_key = key.replace('adapter_layer', 'adapter_layer_dict.adapter_0')
                   new_ordered_dict[new_key] = value
                 else:
                   new_ordered_dict[key] = value
             
             model.load_state_dict(new_ordered_dict,strict=False)
        else: #new checkpoints with new code, that can be loaded directly.
              model.load_state_dict(model_checkpoint['state_dict1'], strict=False)
    elif 'model_state_dict' in model_checkpoint:
          model.load_state_dict(model_checkpoint['model_state_dict'], strict=False)
    


def load_checkpoints(configs, optimizer, scheduler, logging, net):
    """
    Load saved checkpoints from a previous training session.

    Args:
        configs: A python box object containing the configuration options.
        optimizer (Optimizer): The optimizer to resume training with.
        scheduler (Scheduler): The learning rate scheduler to resume training with.
        logging (Logger): The logger to use for logging messages.
        net (nn.Module): The neural network model to load the saved checkpoints into.

    Returns:
        tuple: A tuple containing the loaded neural network model and the epoch to start training from.
    """
    start_epoch = 1

    # If the 'resume' flag is True, load the saved model checkpoints.
    if configs.resume.enable:
        model_checkpoint = torch.load(configs.resume.resume_path, map_location='cpu')
        #net.load_state_dict(model_checkpoint['state_dict1'], strict=False)
        if 'state_dict1' in model_checkpoint:
            #to load old checkpoints that saved adapter_layer_dict as adapter_layer. 
            from collections import OrderedDict
            if np.sum(["adapter_layer_dict" in key for key in model_checkpoint['state_dict1'].keys()])==0: #using old checkpoints, need to rename the adapter_layer into adapter_layer_dict.adapter_0
                 new_ordered_dict = OrderedDict()
                 for key, value in model_checkpoint['state_dict1'].items():
                     if "adapter_layer_dict" not in key:
                       new_key = key.replace('adapter_layer', 'adapter_layer_dict.adapter_0')
                       new_ordered_dict[new_key] = value
                     else:
                       new_ordered_dict[key] = value
                 
                 net.load_state_dict(new_ordered_dict,strict=False)
            else: #new checkpoints with new code, that can be loaded directly.
                  net.load_state_dict(model_checkpoint['state_dict1'], strict=False)
        elif 'model_state_dict' in model_checkpoint:
              net.load_state_dict(model_checkpoint['model_state_dict'], strict=False)
        
        logging.info(f'model checkpoint is loaded from: {configs.resume.resume_path}')
        # If the saved checkpoint contains the optimizer and scheduler states and the epoch number,
        # resume training from the last saved epoch.
        if 'optimizer_state_dict' in model_checkpoint and 'scheduler_state_dict' in model_checkpoint and 'epoch' in model_checkpoint:
            if not configs.resume.restart_optimizer:
                optimizer.load_state_dict(model_checkpoint['optimizer_state_dict'])
                logging.info('Optimizer is loaded to resume training!')

                scheduler.load_state_dict(model_checkpoint['scheduler_state_dict'])
                logging.info('Scheduler is loaded to resume training!')
                start_epoch = model_checkpoint['epoch'] + 1

    # Return the loaded model and the epoch to start training from.
    return net, start_epoch


def save_checkpoint(epoch: int, model_path: str, tools: dict, accelerator: Accelerator):
    """
    Save the model checkpoints during training.

    Args:
        epoch (int): The current epoch number.
        model_path (str): The path to save the model checkpoint.
        tools (dict): A dictionary containing the necessary tools for saving the model checkpoints.
        accelerator (Accelerator): Accelerator object.

    Returns:
        None
    """
    # # Set the path to save the model checkpoint.
    # model_path = os.path.join(tools['result_path'], 'checkpoints', f'checkpoint_{epoch}.pth')

    # Save the model checkpoint.
    torch.save({
        'epoch': epoch,
        'model_state_dict': accelerator.unwrap_model(tools['net'].state_dict()),
        'optimizer_state_dict': accelerator.unwrap_model(tools['optimizer'].state_dict()),
        'scheduler_state_dict': accelerator.unwrap_model(tools['scheduler'].state_dict()),
    }, model_path)


def test_gpu_cuda():
    print('Testing gpu and cuda:')
    print('\tcuda is available:', torch.cuda.is_available())
    print('\tdevice count:', torch.cuda.device_count())
    print('\tcurrent device:', torch.cuda.current_device())
    print(f'\tdevice:', torch.cuda.device(0))
    print('\tdevice name:', torch.cuda.get_device_name(), end='\n\n')


def prepare_tensorboard(result_path):
    train_path = os.path.join(result_path, 'train')
    val_path = os.path.join(result_path, 'val')
    Path(train_path).mkdir(parents=True, exist_ok=True)
    Path(val_path).mkdir(parents=True, exist_ok=True)

    train_log_path = os.path.join(train_path, 'tensorboard')
    train_writer = SummaryWriter(train_log_path)

    val_log_path = os.path.join(val_path, 'tensorboard')
    val_writer = SummaryWriter(val_log_path)

    return train_writer, val_writer


def get_dummy_logging():
    logger = log.getLogger(__name__)
    logger.addHandler(log.NullHandler())
    return logger


def get_logging_old(result_path):
    log.basicConfig(filename=os.path.join(result_path, "logs.txt"),
                    format='%(asctime)s - %(message)s',
                    filemode='a')
    log.getLogger().setLevel(log.INFO)
    log.getLogger().addHandler(log.StreamHandler())
    return log


def get_logging(result_path):
    logger = log.getLogger(result_path)
    logger.setLevel(log.INFO)

    fh = log.FileHandler(os.path.join(result_path, "logs.txt"))
    formatter = log.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = log.StreamHandler()
    logger.addHandler(sh)

    return logger


def save_model(epoch, model, opt, result_path, scheduler, description='best_model'):
    Path(os.path.join(result_path, 'checkpoints')).mkdir(parents=True, exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'optimizer': opt.state_dict(),
        'lr_scheduler': scheduler.state_dict(),
        'epoch': epoch
    }, os.path.join(result_path, 'checkpoints', description + '.pth'))


def random_pick(input_list, num_to_pick, seed):
    # Set the random seed
    random.seed(seed)

    # Check if num_to_pick is greater than the length of the input_list
    if num_to_pick > len(input_list):
        print("Number to pick is greater than the length of the input list")
        return input_list

    # Use random.sample to pick num_to_pick items from the input_list
    random_items = random.sample(input_list, num_to_pick)

    return random_items


if __name__ == '__main__':
    # For test utils modules
    print('done')
