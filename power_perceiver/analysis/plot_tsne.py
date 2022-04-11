from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pytorch_lightning as pl
import sklearn
import torch
import wandb
from matplotlib import pyplot as plt
from torch import nn

from power_perceiver.analysis.simple_callback import SimpleCallback
from power_perceiver.consts import BatchKey


def plot_tsne_of_pv_system_id_embedding(
    batch: dict[BatchKey, torch.Tensor],
    pv_system_id_embedding: nn.Embedding,
) -> plt.Figure:
    """Plot the t-SNE of the embedding for all PV systems in the batch."""
    pv_system_row_numbers_for_all_examples = []
    num_examples = batch[BatchKey.pv_system_row_number].shape[0]
    for example_idx in range(num_examples):
        row_numbers_for_example = batch[BatchKey.pv_system_row_number][example_idx]
        pv_mask = batch[BatchKey.pv_mask][example_idx]
        row_numbers_for_example = row_numbers_for_example[pv_mask]
        row_numbers_for_example = row_numbers_for_example.tolist()
        pv_system_row_numbers_for_all_examples.extend(row_numbers_for_example)

    pv_system_row_numbers_for_all_examples = np.unique(pv_system_row_numbers_for_all_examples)
    pv_id_embedding = pv_system_id_embedding(pv_system_row_numbers_for_all_examples)
    pv_id_embedding = pv_id_embedding.detach().cpu()

    tsne = sklearn.manifold.TSNE(n_components=2, init="pca", learning_rate="auto")
    tsne = tsne.fit_transform(pv_id_embedding)

    fig, ax = plt.subplots()
    ax.scatter(x=tsne[:, 0], y=tsne[:, 1], alpha=0.8)
    ax.set_title("t-SNE of PV system ID embedding for all examples in batch")

    return fig


@dataclass
class LogTSNEPlot(SimpleCallback):
    query_generator_name: str = "decoder_query_generator"

    def __post_init__(self):
        super().__init__()

    def _on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[dict[str, object]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
        tag: str,
    ) -> None:
        """Called when the training batch ends.

        Args:
            outputs: The output from Model.training_step
            tag: train or validation
        """
        if batch_idx == 0:
            query_generator = getattr(pl_module, self.query_generator_name)
            fig = plot_tsne_of_pv_system_id_embedding(batch, query_generator.pv_system_id_embedding)
            wandb.log(
                {
                    # Need to convert to image to avoid bug in matplotlib to plotly conversion
                    f"{tag}/tsne": wandb.Image(fig),
                    "global_step": trainer.global_step,
                },
            )
            plt.close(fig)