import torch
import random
import numpy as np

from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader

from .utils.attention import scaled_dot_product_attention
from .utils.dataset import LabeledDataset
from .utils.revin import RevIN
from .utils.sam import SAM


class SAMFormerArchitecture(nn.Module):
    def __init__(self, num_channels, seq_len, hid_dim, pred_horizon, use_revin=True):
        super().__init__()
        self.revin = RevIN(num_features=num_channels)
        self.compute_keys = nn.Linear(seq_len, hid_dim)
        self.compute_queries = nn.Linear(seq_len, hid_dim)
        self.compute_values = nn.Linear(seq_len, seq_len)
        self.linear_forecaster = nn.Linear(seq_len, pred_horizon)
        self.use_revin = use_revin

    def forward(self, x, flatten_output=True):
        # RevIN Normalization
        if self.use_revin:
            x_norm = self.revin(x.transpose(1, 2), mode='norm').transpose(1, 2) # (n, D, L)
        else:
            x_norm = x
        # Channel-Wise Attention
        queries = self.compute_queries(x_norm) # (n, D, hid_dim)
        keys = self.compute_keys(x_norm) # (n, D, hid_dim)
        values = self.compute_values(x_norm) # (n, D, L)
        if hasattr(nn.functional, 'scaled_dot_product_attention'):
            att_score = nn.functional.scaled_dot_product_attention(queries, keys, values) # (n, D, L)
        else:
            att_score = scaled_dot_product_attention(queries, keys, values) # (n, D, L)
        out = x_norm + att_score # (n, D, L)
        # Linear Forecasting
        out = self.linear_forecaster(out) # (n, D, H)
        # RevIN Denormalization
        if self.use_revin:
            out = self.revin(out.transpose(1, 2), mode='denorm').transpose(1, 2) # (n, D, H)
        if flatten_output:
            return out.reshape([out.shape[0], out.shape[1]*out.shape[2]])
        else:
            return out


class SAMFormer:
    """
    SAMFormer pytorch trainer implemented in the sklearn fashion
    """
    def __init__(self, device='cuda:0', num_epochs=100, batch_size=256, base_optimizer=torch.optim.Adam,
                 learning_rate=1e-3, weight_decay=1e-5, rho=0.5, use_revin=True, random_state=None):
        self.network = None
        self.criterion = nn.MSELoss()
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.base_optimizer = base_optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.rho = rho
        self.use_revin = use_revin
        self.random_state = random_state

    def fit(self, x, y):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            random.seed(self.random_state)
            np.random.seed(self.random_state)
            torch.cuda.manual_seed_all(self.random_state)

        self.network = SAMFormerArchitecture(num_channels=x.shape[1], seq_len=x.shape[2], hid_dim=16,
                                             pred_horizon=y.shape[1] // x.shape[1], use_revin=self.use_revin)
        self.criterion = self.criterion.to(self.device)
        self.network = self.network.to(self.device)
        self.network.train()

        optimizer = SAM(self.network.parameters(), base_optimizer=self.base_optimizer, rho=self.rho,
                        lr=self.learning_rate, weight_decay=self.weight_decay)

        train_dataset = LabeledDataset(x, y)
        data_loader_train = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        progress_bar = tqdm(range(self.num_epochs))
        for epoch in progress_bar:
            loss_list = []
            for (x_batch, y_batch) in data_loader_train:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                # =============== forward ===============
                out_batch = self.network(x_batch)
                loss = self.criterion(out_batch, y_batch)
                # =============== backward ===============
                if optimizer.__class__.__name__ == 'SAM':
                    loss.backward()
                    optimizer.first_step(zero_grad=True)

                    out_batch = self.network(x_batch)
                    loss = self.criterion(out_batch, y_batch)

                    loss.backward()
                    optimizer.second_step(zero_grad=True)
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                loss_list.append(loss.item())
            # =============== save model / update log ===============
            train_loss = np.mean(loss_list)
            self.network.train()
            progress_bar.set_description("Epoch {:d}: Train Loss {:.4f}".format(epoch, train_loss), refresh=True)
        return

    def forecast(self, x, batch_size=256):
        self.network.eval()
        dataset = torch.utils.data.TensorDataset(torch.tensor(x, dtype=torch.float))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        outs = []
        for _, batch in enumerate(dataloader):
            x = batch[0].to(self.device)
            with torch.no_grad():
                out = self.network(x)
            outs.append(out.cpu())
        outs = torch.cat(outs)
        return outs.cpu().numpy()

    def predict(self, x, batch_size=256):
        return self.forecast(x, batch_size=batch_size)
