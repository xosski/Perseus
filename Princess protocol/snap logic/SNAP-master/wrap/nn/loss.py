'''
This material was prepared as an account of work sponsored by an agency of the
United States Government.  Neither the United States Government nor the United
States Department of Energy, nor Battelle, nor any of their employees, nor any
jurisdiction or organization that has cooperated in the development of these
materials, makes any warranty, express or implied, or assumes any legal
liability or responsibility for the accuracy, completeness, or usefulness or
any information, apparatus, product, software, or process disclosed, or
represents that its use would not infringe privately owned rights.
 
Reference herein to any specific commercial product, process, or service by
trade name, trademark, manufacturer, or otherwise does not necessarily
constitute or imply its endorsement, recommendation, or favoring by the United
States Government or any agency thereof, or Battelle Memorial Institute. The
views and opinions of authors expressed herein do not necessarily state or
reflect those of the United States Government or any agency thereof.
 
                 PACIFIC NORTHWEST NATIONAL LABORATORY
                              operated by
                                BATTELLE
                                for the
                   UNITED STATES DEPARTMENT OF ENERGY
                    under Contract DE-AC05-76RL01830
'''

import torch
from torch.nn import functional as F


class UniversalLoss(torch.nn.Module):
    def __init__(
        self, energy_weight=1.0, forces_weight=1.0, huber_delta=0.01
    ) -> None:
        super().__init__()
        self.huber_delta = huber_delta
        self.huber_loss = torch.nn.HuberLoss(reduction="mean", delta=huber_delta)
        
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.float32),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.float32),
        )


    def forward(self, ref, pred):
        # pred passed in as [E, forces]
        E_loss = self.energy_weight * self.huber_loss(ref.energy.view(-1), pred[0].view(-1))
        F_loss = self.forces_weight * conditional_huber_forces(ref.forces, pred[1], huber_delta=self.huber_delta)
        return E_loss, F_loss

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}"
        )


def conditional_huber_forces(ref, pred, huber_delta):
    # Define the multiplication factors for each condition
    factors = huber_delta * torch.tensor([1.0, 0.7, 0.4, 0.1], dtype=torch.float32)

    # Apply multiplication factors based on conditions
    c1 = torch.norm(ref, dim=-1) < 100
    c2 = (torch.norm(ref, dim=-1) >= 100) & (
        torch.norm(ref, dim=-1) < 200
    )
    c3 = (torch.norm(ref, dim=-1) >= 200) & (
        torch.norm(ref, dim=-1) < 300
    )
    c4 = ~(c1 | c2 | c3)

    se = torch.zeros_like(pred)

    se[c1] = torch.nn.functional.huber_loss(
        ref[c1], pred[c1], reduction="none", delta=factors[0]
    )
    se[c2] = torch.nn.functional.huber_loss(
        ref[c2], pred[c2], reduction="none", delta=factors[1]
    )
    se[c3] = torch.nn.functional.huber_loss(
        ref[c3], pred[c3], reduction="none", delta=factors[2]
    )
    se[c4] = torch.nn.functional.huber_loss(
        ref[c4], pred[c4], reduction="none", delta=factors[3]
    )

    return torch.mean(se)


class L2MAELoss(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        assert reduction in ["mean", "sum"]

    def forward(self, 
                input: torch.Tensor, 
                target: torch.Tensor, 
                **kwargs
               ):
        dists = torch.norm(input - target, p=2, dim=-1)
        if self.reduction == "mean":
            return torch.mean(dists)
        elif self.reduction == "sum":
            return torch.sum(dists)


class AtomwiseL2Loss(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        assert reduction in ["mean", "sum"]

    def forward(self,
                input: torch.Tensor,
                target: torch.Tensor,
                natoms: torch.Tensor,
                **kwargs
               ):
        assert natoms.shape[0] == input.shape[0] == target.shape[0]
        assert len(natoms.shape) == 1  # (nAtoms, )

        dists = torch.norm(input - target, p=2, dim=-1)
        loss = natoms * dists

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)


class WeightedLoss(torch.nn.Module):
    def __init__(self, 
                 energy_loss=L2MAELoss(reduction="mean"), 
                 forces_loss=AtomwiseL2Loss(reduction="mean"),
                 energy_weight=0.1,
                 forces_weight=0.9,
                ):
        super().__init__()
        self.energy_loss = energy_loss
        self.forces_loss = forces_loss
        self.energy_weight = energy_weight
        self.forces_weight = forces_weight

    def forward(self, ref, pred):
        # pred passed in as [E, forces]
        E_loss = self.energy_weight * self.energy_loss(ref.energy.view(-1), pred[0].view(-1))
        F_loss = self.forces_weight * self.forces_loss(ref.forces, pred[1], ref.natoms.view(-1))
        return E_loss, F_loss


class SmoothPinballLoss(torch.nn.Module):
    """
    Smooth version of the pinball loss function.
    Modified from https://github.com/Javicadserres/wind-production-forecast/blob/main/src/model/losses.py

    Parameters
    ----------
    alpha : int
        Smoothing rate.

    Attributes
    ----------
    self.quantiles : torch.tensor
    """
    def __init__(self, alpha=0.001):
        super(SmoothPinballLoss,self).__init__()
        
        self.beta = 1/alpha
        self.quantiles = torch.Tensor([0.1, 0.9])

        
    def forward(self, pred, target):
        """
        Computes the loss for the given prediction.
        """

        error = target - pred

        q_error = self.quantiles.to(error.device).view(-1,1) * error
        
        soft_error = F.softplus(-error, self.beta)

        losses = q_error + soft_error
        loss = torch.mean(torch.sum(losses, dim=1))

        return loss
