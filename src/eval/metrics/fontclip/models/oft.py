from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import Module, Parameter

@dataclass
class OFTConfig:
    r: int = 4
    eps: float = 6e-5
    block_share: bool = False
    is_coft: bool = False
    apply_q: bool = True
    apply_k: bool = True
    apply_v: bool = True
    apply_out: bool = True


def project(Q: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """
    ensure that a given matrix Q stays close to the zero matrix within a specified tolerance

    Parameters
    ----------
    R : torch.Tensor
        the matrix to be projected
    eps : torch.Tensor
        the tolerance

    Returns
    -------
    torch.Tensor
    """
    Z = torch.zeros((Q.size(0), Q.size(0)), dtype=Q.dtype, device=Q.device)
    diff = Q - Z
    norm_diff = torch.norm(diff)
    if norm_diff <= eps:
        return Q
    else:
        return Z + eps * (diff / norm_diff)


def project_batch(Q: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    ensure that a given matrix Q stays close to the zero matrix within a specified tolerance

    Parameters
    ----------
    R : torch.Tensor
        the matrix to be projected
    eps : torch.Tensor
        the tolerance

    Returns
    -------
    torch.Tensor
    """
    # scaling factor for each of the smaller block matrix
    eps = eps * 1 / torch.sqrt(torch.tensor(Q.shape[0]))
    Z = (
        torch.zeros((Q.size(1), Q.size(1)), device=Q.device, dtype=Q.dtype)
        .unsqueeze(0)
        .expand_as(Q)
    )
    diff = Q - Z
    norm_diff = torch.norm(Q - Z, dim=(1, 2), keepdim=True)
    mask = (norm_diff <= eps).bool()
    out = torch.where(mask, Q, Z + eps * (diff / norm_diff))
    return out


class OFTLinearLayer(Module):
    def __init__(
        self,
        in_features,
        out_features,
        block_share=False,
        eps=6e-5,
        r=4,
        is_coft=False,
    ):
        super(OFTLinearLayer, self).__init__()

        # Define the reduction rate:
        self.r = r

        # Check whether to use the constrained variant COFT
        self.is_coft = is_coft

        assert in_features % self.r == 0, "in_features must be divisible by r"

        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer("cross_attention_dim", torch.tensor(in_features))
        self.register_buffer("hidden_size", torch.tensor(out_features))

        self.fix_filt_shape = [in_features, out_features]

        # As for the block sharing, please see the page 5 of the OFT paper (https://arxiv.org/pdf/2306.07280.pdf)
        self.block_share = block_share
        # Define the trainable matrix parameter: R
        if self.block_share:
            # Initialized as an identity matrix
            self.Q_shape = [in_features // self.r, in_features // self.r]
            self.Q = Parameter(
                torch.zeros(self.Q_shape[0], self.Q_shape[0]), requires_grad=True
            )

            self.eps = eps * self.Q_shape[0] * self.Q_shape[0]
        else:
            # Initialized as a zero matrix
            self.Q_shape = [self.r, in_features // self.r, in_features // self.r]
            Q = torch.zeros(self.Q_shape[1], self.Q_shape[1])
            Q = torch.stack([Q] * self.r)
            self.Q = Parameter(Q, requires_grad=True)
            self.eps = eps * self.Q_shape[1] * self.Q_shape[1]

    def forward(self, x, proj_weight, proj_bias=None):
        orig_dtype = x.dtype
        dtype = self.Q.dtype

        if self.block_share:
            if self.is_coft:
                with torch.no_grad():
                    self.Q.copy_(project(self.Q, eps=self.eps))
            orth_rotate = self.cayley(self.Q)
        else:
            if self.is_coft:
                with torch.no_grad():
                    self.Q.copy_(project_batch(self.Q, eps=self.eps))
            orth_rotate = self.cayley_batch(self.Q)

        # Block-diagonal parametrization
        block_diagonal_matrix = self.block_diagonal(orth_rotate)

        # fix filter
        fix_filt = proj_weight
        fix_filt = torch.transpose(fix_filt, 0, 1)
        filt = torch.mm(block_diagonal_matrix, fix_filt.to(dtype))
        filt = torch.transpose(filt, 0, 1)

        # Apply the trainable identity matrix
        out = nn.functional.linear(
            input=x.to(orig_dtype), weight=filt.to(orig_dtype), bias=proj_bias
        )

        return out

    def cayley(self, data):
        r, c = list(data.shape)
        # Ensure the input matrix is skew-symmetric: Q = - Q^T
        skew = 0.5 * (data - data.t())
        I = torch.eye(r, device=data.device)
        # Perform the Cayley parametrization
        # Be carefule that skew = - Q
        R = torch.mm(I - skew, torch.inverse(I + skew))

        return R

    def cayley_batch(self, data):
        b, r, c = data.shape
        # Ensure the input matrix is skew-symmetric: Q = - Q^T
        skew = 0.5 * (data - data.transpose(1, 2))
        # I = torch.eye(r, device=data.device).unsqueeze(0).repeat(b, 1, 1)
        I = torch.eye(r, device=data.device).unsqueeze(0).expand(b, r, c)

        # Perform the Cayley parametrization
        # Be carefule that skew = - Q
        R = torch.bmm(I - skew, torch.inverse(I + skew))

        return R

    def block_diagonal(self, R):
        if len(R.shape) == 2:
            # Create a list of R repeated block_count times
            blocks = [R] * self.r
        else:
            # Create a list of R slices along the third dimension
            blocks = [R[i, ...] for i in range(self.r)]

        # Use torch.block_diag to create the block diagonal matrix
        A = torch.block_diag(*blocks)

        return A

