import torch

from src.attacks.base_attacks import BaseIterativeAttack
from src.attacks.procedures import BatchIterativeAttack
from .regularizers import reg_disc, reg_neigh


class SimBABinary(BaseIterativeAttack, BatchIterativeAttack):
    def __init__(self, model, criterion, eps, n_steps, estimator, device="cpu", **kwargs):
        super().__init__(model, criterion, eps, n_steps, device=device)
        BaseIterativeAttack.__init__(self, model=model, n_steps=n_steps)
        BatchIterativeAttack.__init__(self, estimator=estimator)
        self.criterion = criterion
        self.criterion.reduction='none'
        self.is_regularized = False

        self.eps = eps

    def get_loss(self, X: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        self.model.zero_grad()
        y_pred = self.model(X)
        loss = self.criterion(y_pred, y_true)
        return loss

    def generate_changes(self, X):
        random_indices = torch.randint(0, X.shape[1], (X.shape[0], 1, 1)).to(self.device)

        ones = torch.ones_like(X)
        X_minus = X.scatter_add(1, random_indices, -self.eps * ones)
        X_plus = X.scatter_add(1, random_indices, self.eps * ones)

        return X_minus, X_plus


    def step(self, X, y_true):
        X_minus, X_plus = self.generate_changes(X)

        y_pred = self.model(X)
        y_pred_minus = self.model(X_minus)
        y_pred_plus = self.model(X_plus)

        data_all = torch.cat([X, X_minus, X_plus], dim=-1)

        losses = torch.stack([
            self.criterion(y_pred, y_true), 
            self.criterion(y_pred_minus, y_true), 
            self.criterion(y_pred_plus, y_true)], dim=0)

        ind = losses.argmax(0)
        res = torch.zeros(*data_all.shape[:-1], 1).to(self.device)

        for i, row in enumerate(data_all):
            res[i, :, :] = row[:, ind[i].item()].unsqueeze(-1)

        return res


class SimBABinaryDiscReg(SimBABinary):
    def __init__(
        self, model, criterion, eps, n_steps, alpha, disc_models, device="cpu", use_sigmoid=False,
    ):
        super().__init__(model, criterion, eps, n_steps, device=device)
        self.use_sigmoid = use_sigmoid
        self.alpha = alpha
        self.disc_models = disc_models

    def step(self, X, y_true):
        X_minus, X_plus = self.generate_random_changes(X)

        y_pred = self.model(X)
        y_pred_minus = self.model(X_minus)
        y_pred_plus = self.model(X_plus)

        X_all = torch.cat([X, X_minus, X_plus]).view(3, *X.shape)
        losses = torch.stack(
            [
                self.criterion(y_pred, y_true) - self.alpha * reg_disc(X, self.disc_models, self.use_sigmoid),
                self.criterion(y_pred_minus, y_true) - self.alpha * reg_disc(X_minus, self.disc_models, self.use_sigmoid),
                self.criterion(y_pred_plus, y_true) - self.alpha * reg_disc(X_plus, self.disc_models, self.use_sigmoid),
            ],
            dim=0,
        )

        max_loss_indices = torch.argmax(losses, dim=0)
        max_loss_indices = (
            max_loss_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, X.shape[-1])
        )

        X_adv = torch.gather(X_all, 0, max_loss_indices).squeeze(0)
        return X_adv

def govn():
    pass
