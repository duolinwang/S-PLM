import os
import numpy as np
import yaml
import argparse
import torch
import torchmetrics
from time import time, sleep
from tqdm import tqdm
from utils import load_configs, test_gpu_cuda, prepare_tensorboard, prepare_optimizer, save_checkpoint, \
    get_logging, load_checkpoints, prepare_saving_dir, focal_loss
from data import prepare_dataloaders_ptm
from model import prepare_models_secondary_structure_ptm
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed


def remove_label(tensor, output, label):
    mask = tensor != label
    return tensor[mask], output[mask]


def train(epoch, accelerator, dataloader, tools, global_step, tensorboard_log):
    # Initialize metrics
    accuracy = torchmetrics.Accuracy(task="binary")
    precision = torchmetrics.Precision(task="binary")
    recall = torchmetrics.Recall(task="binary")
    auc = torchmetrics.AUROC(task="binary")
    average_precision = torchmetrics.AveragePrecision(task="binary")
    f1_score = torchmetrics.F1Score(task="binary")
    mcc = torchmetrics.MatthewsCorrCoef(task="binary")

    accuracy.to(accelerator.device)
    precision.to(accelerator.device)
    recall.to(accelerator.device)
    auc.to(accelerator.device)
    average_precision.to(accelerator.device)
    f1_score.to(accelerator.device)
    mcc.to(accelerator.device)

    tools["optimizer"].zero_grad()

    epoch_loss = 0
    train_loss = 0
    counter = 0

    progress_bar = tqdm(range(global_step, int(np.ceil(len(dataloader) / tools['accum_iter']))),
                        disable=not accelerator.is_local_main_process, leave=False)
    progress_bar.set_description("Steps")

    for i, data in enumerate(dataloader):
        with accelerator.accumulate(tools['net']):
            sequences, labels, masks, indices = data

            outputs = tools['net'](sequences)

            batch_labels = labels[masks]

            preds = outputs[masks][:, 1]
            losses = tools['loss_function'](preds.to(torch.float64), batch_labels.to(torch.float64), alpha=0.8,
                                            reduction="none", gamma=2)

            # classification loss
            loss = torch.mean(losses)

            # Gather the losses across all processes for logging (if we use distributed training).
            avg_loss = accelerator.gather(loss.repeat(tools["train_batch_size"])).mean()
            train_loss += avg_loss.item() / tools['accum_iter']

            accuracy.update(accelerator.gather(preds).detach(), accelerator.gather(batch_labels).detach())
            precision.update(accelerator.gather(preds).detach(), accelerator.gather(batch_labels).detach())
            recall.update(accelerator.gather(preds).detach(), accelerator.gather(batch_labels).detach())
            auc.update(accelerator.gather(preds).detach(), accelerator.gather(batch_labels).detach())
            average_precision.update(accelerator.gather(preds).detach(),
                                     accelerator.gather(batch_labels.to(torch.int32)).detach())
            f1_score.update(accelerator.gather(preds).detach(), accelerator.gather(batch_labels).detach())
            mcc.update(accelerator.gather(preds).detach(), accelerator.gather(batch_labels).detach())

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(tools['net'].parameters(), tools['grad_clip'])

            tools['optimizer'].step()
            tools['scheduler'].step()
            tools['optimizer'].zero_grad()

        if accelerator.sync_gradients:
            if tensorboard_log:
                tools['train_writer'].add_scalar('step loss', train_loss, global_step)
                tools['train_writer'].add_scalar('learning rate', tools['optimizer'].param_groups[0]['lr'], global_step)

            progress_bar.update(1)
            global_step += 1
            accelerator.log({"train_loss": train_loss, 'lr': tools['optimizer'].param_groups[0]['lr']},
                            step=global_step)

            counter += 1
            epoch_loss += train_loss
            train_loss = 0

        logs = {"step_loss": loss.detach().item(),
                "lr": tools['optimizer'].param_groups[0]['lr']}
        progress_bar.set_postfix(**logs)

    train_loss = epoch_loss / counter
    epoch_acc = accuracy.compute().cpu().item()
    epoch_precision = precision.compute().cpu().item()
    epoch_recall = recall.compute().cpu().item()
    epoch_auc = auc.compute().cpu().item()
    epoch_average_precision = average_precision.compute().cpu().item()
    epoch_f1 = f1_score.compute().cpu().item()
    epoch_mcc = mcc.compute().cpu().item()

    accelerator.log({'train_precision': epoch_precision,
                     'train_recall': epoch_recall,
                     'train_auc': epoch_auc,
                     'train_average_precision': epoch_average_precision,
                     "train_f1": epoch_f1,
                     "train_mcc": epoch_mcc,
                     "train_acc": epoch_acc}, step=epoch)
    if tensorboard_log:
        tools['train_writer'].add_scalar('loss', train_loss, epoch)
        tools['train_writer'].add_scalar('accuracy', epoch_acc, epoch)
        tools['train_writer'].add_scalar('precision', epoch_precision, epoch)
        tools['train_writer'].add_scalar('recall', epoch_recall, epoch)
        tools['train_writer'].add_scalar('auc', epoch_auc, epoch)
        tools['train_writer'].add_scalar('average_precision', epoch_average_precision, epoch)
        tools['train_writer'].add_scalar('f1', epoch_f1, epoch)
        tools['train_writer'].add_scalar('mcc', epoch_mcc, epoch)

    # Reset metrics at the end of epoch
    accuracy.reset()
    precision.reset()
    recall.reset()
    auc.reset()
    average_precision.reset()
    f1_score.reset()
    mcc.reset()

    return train_loss, epoch_acc, epoch_precision, epoch_recall, epoch_auc, epoch_f1, epoch_mcc, epoch_average_precision


def valid(epoch, accelerator, dataloader, tools, tensorboard_log):
    tools['net'].eval()

    # Initialize metrics
    accuracy = torchmetrics.Accuracy(task="binary")
    precision = torchmetrics.Precision(task="binary")
    recall = torchmetrics.Recall(task="binary")
    auc = torchmetrics.AUROC(task="binary")
    average_precision = torchmetrics.AveragePrecision(task="binary")
    positive_f1_score = torchmetrics.F1Score(task="binary")
    mcc = torchmetrics.MatthewsCorrCoef(task="binary")

    accuracy.to(accelerator.device)
    precision.to(accelerator.device)
    recall.to(accelerator.device)
    auc.to(accelerator.device)
    average_precision.to(accelerator.device)
    positive_f1_score.to(accelerator.device)
    mcc.to(accelerator.device)

    counter = 0

    progress_bar = tqdm(range(len(dataloader)),
                        disable=not accelerator.is_local_main_process, leave=False)
    progress_bar.set_description("Steps")

    valid_loss = 0
    for i, data in enumerate(dataloader):
        sequences, labels, masks, indices = data

        with torch.inference_mode():
            outputs = tools['net'](sequences)

            batch_labels = labels[masks]

            preds = outputs[masks][:, 1]
            losses = tools['loss_function'](preds.to(torch.float64), batch_labels.to(torch.float64), alpha=0.8,
                                            reduction="none", gamma=2)

            loss = torch.mean(losses)

            accuracy.update(accelerator.gather(preds).detach(), accelerator.gather(batch_labels).detach())
            precision.update(accelerator.gather(preds).detach(), accelerator.gather(batch_labels).detach())
            recall.update(accelerator.gather(preds).detach(), accelerator.gather(batch_labels).detach())
            auc.update(accelerator.gather(preds).detach(), accelerator.gather(batch_labels).detach())
            average_precision.update(accelerator.gather(preds).detach(),
                                     accelerator.gather(batch_labels.to(torch.int32)).detach())
            positive_f1_score.update(accelerator.gather(preds).detach(), accelerator.gather(batch_labels).detach())
            mcc.update(accelerator.gather(preds).detach(), accelerator.gather(batch_labels).detach())

        counter += 1
        valid_loss += loss.data.item()

        progress_bar.update(1)
        logs = {"step_loss": loss.detach().item(),
                "lr": tools['optimizer'].param_groups[0]['lr']}

        progress_bar.set_postfix(**logs)

    valid_loss = valid_loss / counter

    epoch_acc = accuracy.compute().cpu().item()
    epoch_precision = precision.compute().cpu().item()
    epoch_recall = recall.compute().cpu().item()
    epoch_auc = auc.compute().cpu().item()
    epoch_average_precision = average_precision.compute().cpu().item()
    epoch_positive_f1 = positive_f1_score.compute().cpu().item()
    epoch_mcc = mcc.compute().cpu()

    accelerator.log({
        "valid_f1": epoch_positive_f1,
        "valid_precision": epoch_precision,
        "valid_recall": epoch_recall,
        "valid_auc": epoch_auc,
        "valid_average_precision": epoch_average_precision,
        "valid_acc": epoch_acc,
        "valid_mcc": epoch_mcc,
    },
        step=epoch)
    if tensorboard_log:
        tools['valid_writer'].add_scalar('loss', valid_loss, epoch)
        tools['valid_writer'].add_scalar('accuracy', epoch_acc, epoch)
        tools['valid_writer'].add_scalar('precision', epoch_precision, epoch)
        tools['valid_writer'].add_scalar('recall', epoch_recall, epoch)
        tools['valid_writer'].add_scalar('auc', epoch_auc, epoch)
        tools['valid_writer'].add_scalar('average_precision', epoch_average_precision, epoch)
        tools['valid_writer'].add_scalar('f1', epoch_positive_f1, epoch)
        tools['valid_writer'].add_scalar('mcc', epoch_mcc, epoch)

    # Reset metrics at the end of epoch
    accuracy.reset()
    precision.reset()
    recall.reset()
    auc.reset()
    average_precision.reset()
    positive_f1_score.reset()
    mcc.reset()

    return valid_loss, epoch_acc, epoch_precision, epoch_recall, epoch_auc, epoch_positive_f1, epoch_mcc, epoch_average_precision


def main(args, dict_config, config_file_path):
    configs = load_configs(dict_config, args)
    # overwrite result_path and result_path if set through command

    if type(configs.fix_seed) == int:
        torch.manual_seed(configs.fix_seed)
        torch.random.manual_seed(configs.fix_seed)
        np.random.seed(configs.fix_seed)

    torch.cuda.empty_cache()
    test_gpu_cuda()

    result_path, checkpoint_path = prepare_saving_dir(configs, config_file_path)

    logging = get_logging(result_path)

    accelerator = Accelerator(
        mixed_precision=configs.train_settings.mixed_precision,
        # split_batches=True,
        gradient_accumulation_steps=configs.train_settings.grad_accumulation,
        dispatch_batches=False
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initialize automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("accelerator_tracker", config=None)

    dataloaders_dict = prepare_dataloaders_ptm(configs)
    logging.info('preparing dataloaders are done')

    net = prepare_models_secondary_structure_ptm(configs, logging)
    logging.info('preparing model is done')

    optimizer, scheduler = prepare_optimizer(net, configs, len(dataloaders_dict["train"]), logging)
    logging.info('preparing optimizer is done')

    net, start_epoch = load_checkpoints(configs, optimizer, scheduler, logging, net)

    dataloaders_dict["train"], dataloaders_dict["valid"], dataloaders_dict["test"] = accelerator.prepare(
        dataloaders_dict["train"],
        dataloaders_dict["valid"],
        dataloaders_dict["test"]
    )

    net, optimizer, scheduler = accelerator.prepare(
        net, optimizer, scheduler
    )

    # initialize tensorboards
    train_writer, valid_writer = prepare_tensorboard(result_path)

    tools = {
        'net': net,
        'train_device': configs.train_settings.device,
        'valid_device': configs.valid_settings.device,
        'train_batch_size': configs.train_settings.batch_size,
        'valid_batch_size': configs.valid_settings.batch_size,
        'optimizer': optimizer,
        'mixed_precision': configs.train_settings.mixed_precision,
        'train_writer': train_writer,
        'valid_writer': valid_writer,
        'accum_iter': configs.train_settings.grad_accumulation,
        'loss_function': focal_loss,
        'grad_clip': configs.optimizer.grad_clip_norm,
        'checkpoints_every': configs.checkpoints_every,
        'scheduler': scheduler,
        'result_path': result_path,
        'checkpoint_path': checkpoint_path,
        'logging': logging,
        'num_classes': configs.encoder.num_classes
    }

    logging.info(f'number of train steps per epoch: {np.ceil(len(dataloaders_dict["train"]) / tools["accum_iter"])}')
    logging.info(f'number of valid steps per epoch: {len(dataloaders_dict["valid"])}')
    logging.info(f'number of test steps per epoch: {len(dataloaders_dict["test"])}')

    best_valid_acc = 0
    best_valid_positive_f1 = 0
    global_step = 0
    for epoch in range(start_epoch, configs.train_settings.num_epochs + 1):
        tools['epoch'] = epoch
        start_time = time()
        train_loss, train_acc, train_precision, train_recall, train_auc, train_f1, train_mcc, train_ap = train(
            epoch, accelerator,
            dataloaders_dict["train"], tools, global_step,
            configs.tensorboard_log
        )
        end_time = time()
        if accelerator.is_main_process:
            logging.info(f'epoch {epoch} - time {np.round(end_time - start_time, 2)}s, '
                         f'train loss {np.round(train_loss, 4)}, train acc {np.round(train_acc, 4)}, '
                         f'train precision {np.round(train_precision, 4)}, '
                         f'train recall {np.round(train_recall, 4)}, '
                         f'train auc {np.round(train_auc, 4)}, '
                         f'train average precision {np.round(train_ap, 4)}, '
                         f'train f1 {np.round(train_f1, 4)}, '
                         f'train mcc {np.round(train_mcc, 4):.4f}')

        if epoch % configs.valid_settings.do_every == 0 and epoch != 0:
            start_time = time()
            valid_loss, valid_acc, valid_precision, valid_recall, valid_auc, valid_positive_f1, valid_mcc, valid_ap = valid(
                epoch, accelerator, dataloaders_dict["valid"],
                tools, configs.tensorboard_log
            )
            end_time = time()
            if accelerator.is_main_process:
                logging.info(
                    f'evaluation - time {np.round(end_time - start_time, 2)}s, '
                    f'valid loss {np.round(valid_loss, 4)}, valid acc {np.round(valid_acc, 4)}, '
                    f'valid precision {np.round(valid_precision, 4)}, '
                    f'valid recall {np.round(valid_recall, 4)}, '
                    f'valid auc {np.round(valid_auc, 4)}, '
                    f'valid average precision {np.round(valid_ap, 4)}, '
                    f'valid f1 {np.round(valid_positive_f1, 4)}, '
                    f'valid mcc {np.round(valid_mcc, 4):.4f}'
                )

            if valid_positive_f1 > best_valid_positive_f1:
                best_valid_acc = valid_acc
                best_valid_positive_f1 = valid_positive_f1
                # Set the path to save the model checkpoint.
                model_path = os.path.join(tools['result_path'], 'checkpoints', f'best_model.pth')
                accelerator.wait_for_everyone()
                save_checkpoint(epoch, model_path, tools, accelerator)

        if epoch % configs.checkpoints_every == 0:
            # Set the path to save the model checkpoint.
            model_path = os.path.join(tools['result_path'], 'checkpoints', f'checkpoint_{epoch}.pth')
            accelerator.wait_for_everyone()
            save_checkpoint(epoch, model_path, tools, accelerator)

    if accelerator.is_main_process:
        logging.info(f'\nbest valid acc: {np.round(best_valid_acc, 4)}')
        logging.info(f'best valid positive f1: {np.round(best_valid_positive_f1, 4)}')

    train_writer.close()
    valid_writer.close()

    # pause 20 second to make sure the best validation checkpoint is ready on the disk
    sleep(20)

    model_path = os.path.join(tools['result_path'], 'checkpoints', 'best_model.pth')
    model_checkpoint = torch.load(model_path, map_location='cpu')
    net.load_state_dict(model_checkpoint['model_state_dict'])

    start_time = time()
    test_loss, test_acc, test_precision, test_recall, test_auc, test_positive_f1, test_mcc, test_ap = valid(
        0, accelerator, dataloaders_dict["test"], tools,
        tensorboard_log=False
    )
    end_time = time()
    if accelerator.is_main_process:
        logging.info(
            f'\ntest - time {np.round(end_time - start_time, 2)}s, '
            f'test loss {np.round(test_loss, 4)}')
        logging.info(f'test acc: {np.round(test_acc, 4)}')
        logging.info(f'test precision: {np.round(test_precision, 4)}')
        logging.info(f'test recall: {np.round(test_recall, 4)}')
        logging.info(f'test auc: {np.round(test_auc, 4)}')
        logging.info(f'test average precision: {np.round(test_ap, 4)}')
        logging.info(f'test positive f1: {np.round(test_positive_f1, 4)}')
        logging.info(f'test mcc: {np.round(test_mcc, 4):.4f}')

    accelerator.end_training()
    accelerator.free_memory()
    del tools, net, dataloaders_dict, accelerator, optimizer, scheduler
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a classification model using esm")
    parser.add_argument("--config_path", "-c", help="The location of config file", default='./config_ptm.yaml')
    parser.add_argument("--result_path", default=None,
                        help="result_path, if setted by command line, "
                             "overwrite the one in config.yaml, by default is None")
    parser.add_argument("--resume_path", default=None,
                        help="if set, overwrite the one in config.yaml, by default is None")
    parser.add_argument("--num_end_adapter_layers", default=None, help="num_end_adapter_layers")
    parser.add_argument("--module_type", default=None, help="module_type for adapterh")
    args = parser.parse_args()
    config_path = args.config_path

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    main(args, config_file, config_path)
    print('done!')