import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from dgl import DGLGraph
from dgl.data import citation_graph as citegrh


class GATModel(pl.LightningModule):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads):
        super(GATModel, self).__init__()
        self.net = GAT(g,
                       in_dim=in_dim,
                       hidden_dim=hidden_dim,
                       out_dim=out_dim,
                       num_heads=num_heads)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)

    def forward(self, features):
        return self.net(features)

    def training_step(self, batch, batch_idx):
        features, labels, mask = batch
        logits = self.forward(features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[mask], labels[mask])
        return loss

    def configure_optimizers(self):
        return self.optimizer


def load_cora_data():
    data = citegrh.load_cora()
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    mask = torch.ByteTensor(data.train_mask)
    g = DGLGraph(data.graph)
    return g, features, labels, mask


def main():
    g, features, labels, mask = load_cora_data()

    # Create the GATModel instance
    model = GATModel(g,
                     in_dim=features.size()[1],
                     hidden_dim=8,
                     out_dim=7,
                     num_heads=2)

    # Create PyTorch Lightning Trainer
    trainer = pl.Trainer(max_epochs=30)

    # Train the model
    trainer.fit(model)

    # Optional: You can print the model summary after training
    print(model.net)


if __name__ == "__main__":
    main()
