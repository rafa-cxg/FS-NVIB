#
# Copyright (c) 2022 Idiap Research Institute, http://www.idiap.ch/
# Written by Fabio Fehr <fabio.fehr@idiap.ch>
#

import math

import torch
import torch.nn as nn

# Note:
# Ns: Source length
# Nt: target length
# Nl: latent length
# B: batch size
# H: hidden dimension


def eye_scaled_(tensor, scale=1.0):
    with torch.no_grad():
        torch.eye(*tensor.shape, out=tensor, requires_grad=tensor.requires_grad).mul_(scale)
    return tensor


def init_vector_(tensor, init_vector):
    with torch.no_grad():
        tensor.copy_(init_vector)
    return tensor


class Quadratic(torch.nn.Module):
    def __init__(self, size_in, size_out):
        """
        In the constructor we instantiate three parameters and assign them as
        member parameters.
        """
        super().__init__()

        self.linear = torch.nn.Linear(size_in, size_out)
        self.quadratic = torch.nn.Linear(size_in, size_out, bias=False)

        self.bias = self.linear.bias
        self.weight_linear = self.linear.weight
        self.weight_quadratic = self.quadratic.weight

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        return self.linear(x) + self.quadratic(x**2)


class Exponential(nn.Module):
    """
    Simple exponential activation function
    """

    def __init__(self):
        super().__init__()
        self.activation = torch.nn.ELU()

    def forward(self, x):
        #exp_values_clamped = torch.exp(x)*0.0001
        # exp_values_clamped = torch.clamp(torch.exp(x), max=1e16) #*0.00001
        # negative_indices = x <= 0
        # value = self.activation(x)
        # negative_indices = value <= 0
        # value[negative_indices]+=1
        # upper_limit = 1e8
        # out=torch.exp(x)
        # max_exp_x = out.max()
        # if max_exp_x > upper_limit:
        #     scaled_exp_x = out / max_exp_x * upper_limit
        # else:
        #
        #     scaled_exp_x = out
        # return  scaled_exp_x  #exp_values_clamped
        return torch.exp(x)


class Nvib(nn.Module):
    """
    A Nonparameteric variational information bottleneck layer
    """

    def __init__(
        self,
        size_in,
        size_out,
        prior_mu=None,
        prior_var=None,
        prior_log_alpha=None,
        prior_log_alpha_stdev=None,
        delta=1,
        nheads=1,
        alpha_tau=None,
        stdev_tau=None,
        mu_tau=None,
        learnable_prior=False,
        **kwargs
    ):
        super().__init__()

        # Dimensionality:
        # size_in: P
        # size_out: P
        # nheads: H
        # head_dim: P/H = D
        # length: Ns
        # latent_length: Nl
        # batch: B

        # Prior mean [P]
        if prior_mu is not None:
            self.prior_mu = nn.Parameter(prior_mu, requires_grad=learnable_prior)
        else:
            self.prior_mu = nn.Parameter(torch.zeros(size_in), requires_grad=learnable_prior)

        # Prior variance [P]
        if prior_var is not None:
            self.prior_var = nn.Parameter(prior_var, requires_grad=False)
        else:
            self.prior_var = nn.Parameter(torch.ones(size_in), requires_grad=False)

        # Prior log alpha [1]
        if prior_log_alpha is not None:
            self.prior_log_alpha = nn.Parameter(prior_log_alpha, requires_grad=learnable_prior)
        else:
            self.prior_log_alpha = nn.Parameter(torch.zeros(1), requires_grad=learnable_prior)

        # Prior log alpha standard deviation (important for initialisation) [1]
        if prior_log_alpha_stdev is not None:
            self.prior_log_alpha_stdev = nn.Parameter(prior_log_alpha_stdev, requires_grad=False)
        else:
            self.prior_log_alpha_stdev = nn.Parameter(torch.ones(1), requires_grad=False)

        # Conditional prior delta for the dirichlet KL divergence
        self.delta = float(delta)

        # Layers for parameters
        self.size_in = size_in
        self.size_out = size_out
        self.d = int(size_in / nheads)  # dimension of the head
        self.alpha_activation = Exponential()  # projection for alphas
        self.mu_proj = nn.Linear(size_in, size_out)  # Project to mean
        self.logvar_proj = nn.Linear(size_in, size_out)  # Project log variance
        self.alpha_proj = Quadratic(size_in, 1)  # Project to model size
        self.nheads = nheads  # number of heads

        # Initialisation parameters - 0 is the prior 1 is the posterior
        self.alpha_tau = alpha_tau if alpha_tau is not None else 1
        self.stdev_tau = stdev_tau if stdev_tau is not None else 1
        self.mu_tau = mu_tau if mu_tau is not None else 1
        # Initialise the parameters
        self.init_parameters()

    def init_parameters(self):
        """
        Initialise parameters
        """
        # Initialise mu projection
        eye_scaled_(self.mu_proj.weight, self.mu_tau)
        init_vector_(self.mu_proj.bias, self.prior_mu * (1 - self.mu_tau))

        # Initialise logvar projection
        nn.init.constant_(self.logvar_proj.weight, 0)
        init_vector_(
            self.logvar_proj.bias,
            torch.log(
                (torch.sqrt(self.prior_var) * self.stdev_tau)
                ** 2  # Controls the standard deviation
                + torch.finfo(self.prior_var.dtype).tiny
            ),  # nonzero
        )

        # Initialise alpha projection
        nn.init.constant_(self.alpha_proj.weight_quadratic, 1 / (2 * math.sqrt(self.d)))
        nn.init.constant_(self.alpha_proj.weight_linear, 0)
        init_vector_(
            self.alpha_proj.bias,
            self.prior_log_alpha_stdev * (self.alpha_tau),  # Standard deviation of log alpha
        )

    def reparameterize_gaussian(self, mu, logvar, set_mata_training_mode):
        """
        Reparameterise for gaussian
        Train = sample
        Test = mean

        :param mu: means [Nl,B,P]
        :param logvar: logged variances [Nl,B,P]
        :return: z: sample from a gaussian distribution or mean [Nl,B,P]
        """

        if set_mata_training_mode:
            std = torch.exp(0.5 * logvar)  # [Nl,B,P]
            eps = torch.randn_like(std)  # [Nl,B,P]
            z = eps.mul(std).add_(mu)  # [Nl,B,P]
        else:
            z = mu  # [Nl,B,P]
        return z  # [Nl,B,P]

    def reparameterize_dirichlet(self, alpha, mask, set_mata_training_mode):
        """
        Reparameterise for dirichlet
        Train = sample
        Test = mean

        :param alpha: psuedo-counts [B,Nl,1]
        :param mask: Mask for the latent space [B,Nl]
        :return pi: dirichlet probability [B,Nl,1]
        """
        if mask is not None:
            mask = mask.unsqueeze(-1)
            # Sample from the gamma distribution
        if set_mata_training_mode:
            # rsample() can cause NaNs when the alphas are too large and too small.
            #    We keep the proportion and scale it. Also we need to clamp the masked values
            if mask is not None:
                alpha_sum = torch.sum(alpha.masked_fill(mask, 0), dim=1, keepdim=True)
            else:
                alpha_sum = torch.sum(alpha, dim=1, keepdim=True)
            # Get proportions
            alpha = alpha / alpha_sum
            # Clamp the proportion to prevent underflow - Clamp the sum of alpha to prevent overflow.
            alpha = torch.clamp(alpha, min=1e-23) * torch.clamp(alpha_sum, max=1e8)
            # Clamp the masked values!
            if mask is not None:
                alpha.masked_fill_(mask, 1e-15)

            gamma_dist = torch.distributions.Gamma(alpha, torch.ones_like(alpha))
            gammas = gamma_dist.rsample()

            # Testing the alphas don't have noise
        else:
            gammas = alpha

        if mask is not None:
            gammas.masked_fill_(mask, 0)
        normalising_sum = torch.sum(gammas, 1, keepdim=True) + torch.finfo(gammas.dtype).tiny
        pi = torch.div(gammas, normalising_sum)

        return pi



    def kl_gaussian(self, mu, logvar, alpha, mask=None, **kwargs):
        """
        KL Loss for the Gaussian component with expected K
        :param mu: mean [Nl,B,P]
        :param logvar: logged variance [Nl,B,P]
        :param alpha: psuedo count weight [Nl,B,1]
        :param mask: boolean mask [B,Nl]
        :return: KL [B]
        """

        # Scaling
        # Total number of vectors sampled
        if mask is not None:
            k0 = torch.sum(~mask, 1)  # [B]
        else:
            k0 = torch.full((alpha.size(0),), alpha.size(1), device=alpha.device)  # [B]

        # Input length
        n = k0  # / self.kappa  # [B]

        alpha = alpha.masked_fill(mask.unsqueeze(-1), 0) if mask is not None else alpha
        alpha0_q = torch.sum(alpha, dim=1, keepdim=True) + torch.finfo(alpha.dtype).tiny # [B,1]#todo 为什么不用重采样的pi,那个用gamma比alpha更准吧？
        expected_pi = (alpha / alpha0_q).squeeze(-1)  # [B,Nl]

        # KL between univariate Gaussians
        var_ratio = logvar.exp() / self.prior_var
        t1 = (mu - self.prior_mu) ** 2 / self.prior_var
        kl = var_ratio + t1 - 1 - var_ratio.log()
        kl = kl.masked_fill(mask.unsqueeze(-1), 0) if mask is not None else kl

        # Mean over embedding dimension
        kl = torch.mean(kl, -1)  # [B, Nl]

        # Scale and sum over sentence length dimension
        kl = 0.5 * k0 * torch.sum(kl * expected_pi, -1)  # [B]
        kl /= n

        return kl

    def kl_dirichlet(self, alpha, mask=None, **kwargs):
        """
        The regularisation for the dirichlet component with expected K

        :param alpha: k dimensional psuedo counts [B,Nl,1]
        :param mask: boolean mask [B,Nl]
        :return: Kl [B]

        Nota Bene: digamma and lgamma cannot be zero
        """

        # Total number of vectors sampled
        if mask is not None:
            k0 = torch.sum(~mask, 1)  # [B]
        else:
            k0 = torch.full((alpha.size(0),), alpha.size(1), device=alpha.device)  # [B]

        # k0 = 1
        # Input length
        n = k0  # / self.kappa  # [B]

        alpha = alpha.masked_fill(mask.unsqueeze(-1), 0) if mask is not None else alpha
        alpha0_q = torch.sum(alpha, 1).squeeze(-1).to(torch.float64) + torch.finfo(alpha.dtype).tiny # [B] todo 我加了小数避免0
        # Conditional prior with lower bound. Sentence length weighted by delta
        alpha0_p = (
            torch.exp(self.prior_log_alpha).repeat(alpha.size(0)) + self.delta * (n - 1)
        ).to(
            torch.float64
        )  # [B]

        # KL between two dirichlet distributions
        kl = (
            torch.lgamma(alpha0_q)
            - torch.lgamma(alpha0_p)
            + (alpha0_q - alpha0_p) * (-torch.digamma(alpha0_q) + torch.digamma(alpha0_q / k0))
            + k0 * (torch.lgamma(alpha0_p / k0) - torch.lgamma(alpha0_q / k0))
        ) / n

        return kl

    def forward(self, encoder_output, mask=None, set_mata_training_mode=False, **kwargs):
        """
        The latent layer for NVIB. Notice length comes in as NS and exits Nl (Ns+1) for the prior
        :param encoder_output:[B, Ns, P]
        :param mask: [B,Ns] boolean mask. True is padding
        :return: A tuple of outputs (z, mu, logvar, alpha, pi, memory_key_padding_mask)
                z: sample from the gaussian [B,Nl,P]
                pi: sampled from the dirichlet [B,Nl,1]
                mu: means from the latent layer [B,Nl,P]
                logvar: logged variances from the latent layer [B, Nl, P]
                alpha: logged psuedo-counts from the latent layer [B,Nl,heads]
                memory_key_padding_mask: from the latent layer [B,Nl]


        """
        # Get batch size
        B = encoder_output.size(0)

        # Project to mean, log variance and log alpha
        mu = self.mu_proj(encoder_output)
        logvar = self.logvar_proj(encoder_output)
        alpha = self.alpha_activation(self.alpha_proj(encoder_output))

        # Include the priors in the first position of the latent embeddings
        mu = torch.cat((self.prior_mu.repeat(B, 1, 1), mu), 1)
        logvar = torch.cat((torch.log(self.prior_var).repeat(B, 1, 1), logvar), 1)# 197->198
        alpha = torch.cat((torch.exp(self.prior_log_alpha).repeat(B, 1, 1), alpha), 1) #torch.cat((self.alpha_activation(self.prior_log_alpha).repeat(B, 1, 1), alpha), 1)
        mask = (
            torch.cat((torch.zeros((B, 1), dtype=torch.bool), mask), 1)
            if mask is not None
            else None
        )

        # Reparameterise parameters
        z = self.reparameterize_gaussian(mu, logvar, set_mata_training_mode=set_mata_training_mode)
        pi = self.reparameterize_dirichlet(alpha, mask, set_mata_training_mode=set_mata_training_mode)

        return z, pi, mu, logvar, alpha, mask
