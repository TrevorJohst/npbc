from typing import Tuple
import normflows as nf
import torch
from torch import nn
from torch.distributions.wishart import _mvdigamma
import math


class NormalWishart:
    """
    A Normal-Wishart distribution.
    """

    def __init__(
        self,
        mu: torch.Tensor,
        kappa: torch.Tensor,
        W: torch.Tensor,
        nu: torch.Tensor
    ):
        """
        Args:
            mu: Location vector; shape of [... D]
            kappa: Confidence in the mean vector; must be > 0; shape of [... 1]
            W: Scale matrix; shape of [... D D]
            nu: Degrees of freedom, i.e. confidence in scale matrix; must be >= D - 1; shape of [... 1]
        """
        self.d = mu.size(-1)
        self.mu = mu
        self.kappa = kappa
        self.W = W
        self.nu = nu

    @classmethod
    def from_priors(
        cls,
        location: torch.Tensor,
        location_evidence: float,
        ranges: torch.Tensor,
        range_evidence: float,
        uniform: bool = False
    ):
        """
        Constructs a Normal-Wishart distribution with prior beliefs and evidence on those beliefs.

        Args:
            location: The expected location of the distribution; shape of [D]
            location_evidence: Confidence in the location estimate; must be > 0
            ranges: Expected mean centered standard deviation for each dimension; shape of [D]
            range_evidence: Confidence in the range estimates; must be >= D - 1
            uniform: If true, assumes ranges are of a uniform distribution, otherwise uses a normal
        """
        
        # Construct the prior precision matrix
        variances = (ranges**2) / 3.0 if uniform else (ranges / 2.0)**2
        Sigma = torch.diag(variances)
        W = torch.inverse(Sigma) / range_evidence

        return cls(location, torch.tensor(location_evidence, device=location.device), W, torch.tensor(range_evidence, device=location.device))

    def expected_log_likelihood(self, y: torch.Tensor) -> torch.Tensor:
        """
        Computes `E_μ,Λ[log N(y|μ,Λ)]` for a multivariate normal under the normal wishart distribution.
        """

        # Scalar term
        scalar = -self.d / 2 * torch.log(torch.tensor(2*torch.pi, device=y.device))
        
        # E_Λ[log |Λ|] - standard result
        E_logdet = _mvdigamma(self.nu / 2, p=self.d) + self.d * torch.log(torch.tensor(2, device=y.device)) + torch.logdet(self.W)

        # E_μ,Λ[(y - μ)ᵀΛ(y - μ)]
        E_mse = torch.einsum('...i,...ij,...j->...', y - self.mu, self.nu.unsqueeze(-1) * self.W, y - self.mu) + (self.d / self.kappa)

        # E_μ,Λ[log N(y|μ,Λ)] = scalar + 1/2 * E_Λ[log |Λ|] - 1/2 * E_μ,Λ[(y - μ)ᵀΛ(y - μ)]
        return scalar + 0.5 * E_logdet - 0.5 * E_mse
    
    def entropy(self) -> torch.Tensor:
        """
        Computes `H[p(μ,Λ)] = E_μ,Λ[-log p(μ,Λ)]` for this normal wishart distribution.
        """

        # -log(z⁻¹) - the scalar term
        scalar = (
            self.d / 2 * torch.log(self.kappa / (2 * torch.pi))
            + self.nu / 2 * torch.logdet(self.W)
            + self.d * self.nu / 2 * torch.log(torch.tensor(2, device=self.mu.device))
            + torch.special.multigammaln(self.nu / 2, p=self.d)
        )

        # E_Λ[log |Λ|] - standard result
        E_logdet = _mvdigamma(self.nu / 2, p=self.d) + self.d * torch.log(torch.tensor(2, device=self.mu.device)) + torch.logdet(self.W)

        # H[p(μ,Λ)] = scalar - 1/2 * E_Λ[log |Λ|] + d/2 - (k - d - 1)/2 * E_Λ[log |Λ|] + 1/2 * dν
        return scalar - 0.5 * E_logdet + self.d / 2 - (self.nu - self.d - 1) / 2 * E_logdet + 0.5 * self.d * self.nu
    
    def maximum_a_posteriori(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the parameters μ and Λ of the MAP estimate after the posterior update.
        """
        return(
            self.mu,
            (self.nu - self.d - 1).view(-1, 1, 1) * self.W
        )


class NaturalPosteriorNetworkHead(nn.Module):
    """
    A head for a natural posterior network to be implemented on any n-dimensional regression task.
    (NatPN paper: https://arxiv.org/abs/2105.04471, code: https://github.com/borchero/natural-posterior-network)
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        prior: NormalWishart,
        normalizing_flow: nf.NormalizingFlow,
    ):
        """
        Args:
            latent_dim: Dimension of the latent vector input into the head.
            output_dim: The action output dimension.
            prior: A NormalWishart distribution instantiated with the parameters for your desired prior distribution.
            normalizing_flow: The invertible flow that yields `p(x_i)` where `x_i` is the latent representation.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.normalizing_flow = normalizing_flow

        # Output maps from latent to an n-dimensional normal
        # NOTE: Using an S^3 distribution for orientations may be more stable
        cholesky_entries = (output_dim * (output_dim + 1)) // 2
        self.linear = nn.Linear(latent_dim, output_dim + cholesky_entries)
        self.register_buffer('cholesky_indices', torch.tril_indices(output_dim, output_dim))
        self.register_buffer('diag_entries', torch.arange(self.output_dim))

        # Store prior distribution parameters
        self.register_buffer('mu_0',    prior.mu)
        self.register_buffer('kappa_0', prior.kappa)
        self.register_buffer('W_0',     prior.W)
        self.register_buffer('nu_0',    prior.nu)

    def forward(self, z: torch.Tensor) -> Tuple[NormalWishart, torch.Tensor]:
        """
        Performs a Bayesian update over the target distribution.

        Args:
            z: The penultimate layer of the BC model - the latent vector. [batch, latent_dim]

        Returns:
            A tuple of the updated Normal-Wishart posterior and the log-probability
            from the normalizing flow.
        """
        batch_size = z.size(0)

        # Predict the parameters of our output distribution
        params = self.linear.forward(z)

        # First n params are the mean
        mu = params[:, :self.output_dim]

        # Remaining params are the Cholesky factorization of the precision (inverse of the covariance)
        cholesky_entries = params[:, self.output_dim:]
        precision_cholesky = torch.zeros(batch_size, self.output_dim, self.output_dim, device=params.device)
        precision_cholesky[:, self.cholesky_indices[0], self.cholesky_indices[1]] = cholesky_entries
        precision_cholesky[:, self.diag_entries, self.diag_entries] = precision_cholesky[:, self.diag_entries, self.diag_entries].exp()
        precision_cholesky = precision_cholesky + torch.eye(self.output_dim, device=params.device) * 1e-6   # Numerical stability

        # Compute the predicted covariance
        covariance = torch.inverse(precision_cholesky @ precision_cholesky.transpose(-2, -1))

        # Normalizing flow to recover the evidence
        log_prob = self.normalizing_flow.forward(z)

        # log_prob = self.normalizing_flow.log_prob(z)

        # z_base, log_det = self.normalizing_flow.forward_and_log_det(z)
        # log_prob = log_det + self.normalizing_flow.q0.log_prob(z_base)

        log_evidence = self._scale_evidence(log_prob)
        evidence = log_evidence.exp()
        
        # Update based on the prior
        posterior = self._prior_update(mu, covariance, evidence)

        # Return the updated posterior and the log-evidence
        return posterior, log_evidence

    def _prior_update(
        self, 
        mu: torch.Tensor, 
        covariance: torch.Tensor, 
        evidence: torch.Tensor
    ) -> NormalWishart:
        """
        Updates the posterior distribution based on the stored prior.

        Args:
            mu: The normal's location. [batch, output_dim]
            covariance: The normal's covariance matrix. [batch, output_dim, output_dim]
            evidence: Predicted evidence for the normal. [batch]

        Returns:
            The updated posterior distribution.
        """
        evidence_b1 = evidence.unsqueeze(-1)
        prior_mu, prior_kappa, prior_W, prior_nu = self.mu_0, self.kappa_0, self.W_0, self.nu_0

        # Compute the updated NW parameters
        # https://en.wikipedia.org/wiki/Normal-Wishart_distribution
        posterior_mu = (prior_kappa * prior_mu + evidence_b1 * mu) / (prior_kappa + evidence_b1)
        posterior_kappa = prior_kappa + evidence_b1
        posterior_W = torch.inverse(
            torch.inverse(prior_W)
            + evidence.view(-1, 1, 1) * covariance 
            + ( (evidence_b1 * prior_kappa) / posterior_kappa ).view(-1, 1, 1) * torch.einsum('bi,bj->bij', mu - prior_mu, mu - prior_mu)
        )
        posterior_nu = prior_nu + evidence_b1

        # Return the updated posterior distribution
        return NormalWishart(posterior_mu, posterior_kappa, posterior_W, posterior_nu)
    
    def _scale_evidence(self, log_prob: torch.Tensor) -> torch.Tensor:
        """
        Scales the evidence in log-space with an exponential certainty budget. (see `Appendix I.6`)

        Args:
            log_probability: The log-probability to scale.

        Returns:
            The scaled and clamped evidence in log-space.
        """
        # Exponential
        # scaled_log_prob = log_prob + self.latent_dim
        # Normal
        scaled_log_prob = log_prob + 0.5 * math.log(4 * math.pi) * self.latent_dim
        return scaled_log_prob + (scaled_log_prob.clamp(min=-30.0, max=30.0) - scaled_log_prob).detach()    