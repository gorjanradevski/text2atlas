import argparse
import json
import logging
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import spacy
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from torch.utils.data import DataLoader, Dataset

from voxel_mapping.anchors import create_ind2anchors, create_ind2centers
from voxel_mapping.losses import OrganDistanceLoss, MinDistanceLoss, BaselineRegLoss
from voxel_mapping.constants import VOXELMAN_CENTER

colors = mcolors.CSS4_COLORS
logging.basicConfig(level=logging.INFO)


@torch.no_grad()
def visualize_mappings(
    samples: List, organs_dir_path: str, model: nn.Module, device: torch.device
):

    organ2voxels = json.load(open(os.path.join(organs_dir_path, "organ2voxels.json")))
    organ2ind = json.load(open(os.path.join(organs_dir_path, "organ2ind.json")))
    ind2organ = dict(zip(organ2ind.values(), organ2ind.keys()))

    organ_indices = [sample["organ_indices"] for sample in samples]
    organ_indices = [item for sublist in organ_indices for item in sublist]
    organ_indices = list(set(organ_indices))
    organs = [ind2organ[organ_index] for organ_index in organ_indices]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for i, organ in enumerate(organs):
        points = organ2voxels[organ]
        points = random.sample(points, int(len(points) / 500))
        points = np.array(points)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], marker=".", alpha=0.05)

    for sample in samples:
        sentence_vector = torch.tensor(sample["vector"]).unsqueeze(0).to(device)
        color = colors[list(colors.keys())[np.array(sample["organ_indices"]).sum()]]
        label = "_".join(
            [ind2organ[organ_ind] for organ_ind in sample["organ_indices"]]
        )
        coordinates = model(sentence_vector).cpu().numpy() * np.array(VOXELMAN_CENTER)
        ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            coordinates[:, 2],
            c=color,
            s=100,
            marker="*",
            edgecolor="k",
            label=label,
        )

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())


class Feedforward(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.tanh(output)
        return output


class SentenceVectorDataset(Dataset):
    def __init__(self, samples: List, ind2anchors: Dict):
        self.samples = samples
        self.sentence_vectors, self.mappings, self.organs_indices = ([], [], [])
        for element in self.samples:
            self.sentence_vectors.append(element["vector"])
            self.mappings.append([ind2anchors[ind] for ind in element["organ_indices"]])
            self.organs_indices.append(element["organ_indices"])

    def __len__(self):
        return len(self.sentence_vectors)

    def __getitem__(self, idx: int):
        sentence_vector = torch.tensor(self.sentence_vectors[idx])
        mapping = torch.tensor(self.mappings[idx]) / torch.tensor(VOXELMAN_CENTER)
        num_organs = len(mapping)

        return sentence_vector, mapping, num_organs


def collate_pad_batch(batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
    sentence_vectors, mappings, num_organs = zip(*batch)

    sentence_vectors = torch.stack(sentence_vectors)
    padded_mappings = torch.nn.utils.rnn.pad_sequence(
        mappings, batch_first=True, padding_value=0
    )
    num_organs = torch.tensor([*num_organs])

    return sentence_vectors, padded_mappings, num_organs


def embed_sample_sentences(samples_json_path: str):
    nlp = spacy.load("en_core_web_md")
    samples = json.load(open(samples_json_path))

    for sample in samples:
        sentence = sample["text"]
        doc = nlp(sentence)
        sample["vector"] = doc.vector.tolist()

    return samples


def test_loss_function(
    samples_json_path,
    organs_dir_path,
    num_epochs=10,
    batch_size=8,
    learning_rate=1e-3,
    weight_decay=0,
    loss_type="all_voxels",
):

    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = embed_sample_sentences(samples_json_path)
    organ2ind_path = os.path.join(organs_dir_path, "organ2ind.json")
    organ2voxels_path = os.path.join(organs_dir_path, "organ2voxels.json")
    organ2center_path = os.path.join(organs_dir_path, "organ2center.json")

    input_size = len(samples[0]["vector"])
    hidden_size = input_size // 2
    output_size = 3

    model = Feedforward(input_size, hidden_size, output_size)
    model = model.to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    if loss_type == "all_voxels":
        logging.warning("Using all organ points!")
        ind2anchors = create_ind2anchors(organ2ind_path, organ2voxels_path, 1000)
        criterion = OrganDistanceLoss()
    elif loss_type == "one_voxel":
        logging.warning("Using only one organ center!")
        ind2anchors = create_ind2centers(organ2ind_path, organ2center_path)
        criterion = MinDistanceLoss()
    else:
        logging.warning("Using only one organ center with baseline MSE loss")
        ind2anchors = create_ind2centers(organ2ind_path, organ2center_path)
        criterion = BaselineRegLoss()

    dataset = SentenceVectorDataset(samples, ind2anchors)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        collate_fn=collate_pad_batch,
    )

    for epoch in range(num_epochs):
        logging.info(f"Starting epoch {epoch + 1}...")
        with tqdm(total=len(dataloader)) as pbar:
            for sentence_vectors, true_mappings, num_organs in dataloader:

                optimizer.zero_grad()

                sentence_vectors, true_mappings, num_organs = (
                    sentence_vectors.to(device),
                    true_mappings.to(device),
                    num_organs.to(device),
                )

                mappings = model(sentence_vectors)
                loss = criterion(mappings, true_mappings, num_organs)

                loss.backward()

                optimizer.step()

                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({"Batch loss": loss.item()})
        if not epoch % 20:
            visualize_mappings(samples, organs_dir_path, model, device)
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Testing loss function")
    parser.add_argument(
        "--samples_json_path", type=str, help="Path to json file with test samples"
    )
    parser.add_argument(
        "--organs_dir_path", type=str, help="Path to the directory with organ info"
    )
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay parameter"
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="mse_loss",
        help="The type of loss to use to train the model",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    test_loss_function(
        args.samples_json_path,
        args.organs_dir_path,
        args.num_epochs,
        args.batch_size,
        args.learning_rate,
        args.weight_decay,
        args.loss_type,
    )


if __name__ == "__main__":
    main()
