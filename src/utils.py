import pandas as pd
import torch
import matplotlib.pyplot as plt
import wandb

from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score


def plot_wandb_metrics(run_path: str, metric_keys: list[str], title_suffix: str):
    """
    Builds a loss plot and a plot of other specified metrics for a given W&B run.

    :param run_path: path to the run in the format “entity/project/run_id”
    :param metric_keys: list of metric keys (e.g., [‘eval/accuracy’, ‘eval/f1_macro’])
    :param title_suffix: name for the metrics plot title (e.g., “F1 metrics”)
    """
    api = wandb.Api()
    run = api.run(run_path)
    history = run.history()

    history = history.fillna(method='ffill').fillna(method='bfill')
    history = history.set_index('train/global_step')
    history = history.sort_index()

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    if 'train/loss' in history.columns:
        axs[0].plot(history['train/loss'].dropna(), color='blue', label='Train Loss')
    if 'eval/loss' in history.columns:
        axs[0].plot(history['eval/loss'].dropna(), color='orange', label='Val Loss')
    axs[0].set_xlabel('train/global_step')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Loss')
    axs[0].legend()
    
    if title_suffix is not None:
        for metric in metric_keys:
            if metric in history.columns:
                axs[1].plot(history[metric].dropna(), label=metric)
        axs[1].set_xlabel('train/global_step')
        axs[1].set_ylabel('Metric value')
        axs[1].set_title(f'Plot of metrics: {title_suffix}')
        axs[1].legend()
    
    plt.tight_layout()
    plt.show()


def compute_accuracy(pred):
    """
    Calculates accuracy

    :param pred: predictions
    :return: dictionary with accuracy value
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}


def extract_cls(dataset, model):
    """
    Each element of the dataset is run through a transformer and receives a CLS token representation

    :param dataset: dataset
    :return: CLS token representations and labels
    """
    cls_vecs, labels = [], []
    dataloader = DataLoader(dataset, batch_size=64)
    with torch.no_grad():
        for batch in tqdm(dataloader, leave=False):
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            cls = output.last_hidden_state[:, 0, :].cpu()
            cls_vecs.append(cls)
            labels.append(batch["label_id"])
    return torch.cat(cls_vecs), torch.cat(labels)