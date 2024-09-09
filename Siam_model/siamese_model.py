import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split


class EmbeddingDataset(Dataset):

    def __init__(self, hla_embedding, peptide_embedding) -> None:
        super(EmbeddingDataset).__init__()

        self.hla_embedding = hla_embedding
        self.peptide_embedding = peptide_embedding

    def __len__(self):
        assert len(self.hla_embedding) == len(self.peptide_embedding)
        return len(self.hla_embedding)

    def __getitem__(self, idx):
        return self.hla_embedding[idx], self.peptide_embedding[idx]


class EmbeddingDataModule(pl.LightningDataModule):

    def __init__(self, hla, peptide, batch_size):
        super(EmbeddingDataModule).__init__()
        self.hla = hla
        self.peptide = peptide
        self.batch_size = batch_size

    def setup(self, stage=None):
        full_dataset = EmbeddingDataset(self.hla, self.peptide)

        self.train, self.val = random_split(full_dataset, [int(len(full_dataset) * 0.8),
                                                           len(full_dataset) - int(len(full_dataset) * 0.8)])


    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=4)


def contrastive_loss(vec1, vec2, score_assoc):
    margin = 1.5
    euclidean_distance = (vec1 - vec2).pow(2).sum(1).sqrt()

    return 1 / 2 * (
                score_assoc * euclidean_distance.pow(2) + (1 - score_assoc) * max(0, margin - euclidean_distance) ** 2)


class SiameseModel(nn.Module):
    def __init__(self, input_dim) -> None:
        super(SiameseModel, self).__init__()
        self.input_dim = input_dim

        self.featurizer = nn.Sequential(
            nn.Conv1d(self.input_dim, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten()

        )

        self.conv2d_block = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(1),  # [batch_size, 64, 1, 1]
            nn.Flatten(),
        )

        self.scorecompute_block = nn.Sequential(

            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )


    def forward(self, x1, x2):
        feat_x1, feat_x2 = self.featurizer(x1), self.featurizer(x2)

        print(feat_x1)
        print(feat_x2)
        print(feat_x1.shape)
        print(feat_x2.shape)


        combined_representation = torch.reshape(feat_x1 - feat_x2, (1, 128))


        score = self.scorecompute_block(combined_representation)


        return score

    def training_step(self, batch, ):
        mha, peptide, association = batch

        predicted_association = self.forward(mha, peptide)

        train_loss = contrastive_loss(mha, peptide, association)
        return train_loss





