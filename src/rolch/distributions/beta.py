import numpy as np
import scipy.special as spc
import scipy.stats as st

from rolch.base import Distribution, LinkFunction
from rolch.link import LogLink


class DistributionBeta(Distribution):
    """The Beta Distribution for GAMLSS.

    The distribution function is defined as in GAMLSS as:
    $$ 
    f(y|\mu,\sigma)= \\frac{\Gamma(\\frac{\mu(1-\mu)}{\sigma^2} - 1)}
    {\Gamma(\mu (\\frac{\mu(1-\mu)}{\sigma^2} - 1)) 
    \Gamma((1-\mu) (\\frac{\mu(1-\mu)}{\sigma^2} - 1))}
    x^{\mu (\\frac{\mu(1-\mu)}{\sigma^2} - 1) - 1} 
    (1 - x)^{(1-\mu) (\\frac{\mu(1-\mu)}{\sigma^2} - 1) - 1}
    $$

    with the location and shape parameters $\mu, \sigma > 0$.

    !!! Note
        The function is parameterized as GAMLSS' BE() distribution.

        This parameterization is different to the `scipy.stats.gamma(alpha, loc, scale)` parameterization.

        We can use `DistributionGamma().gamlss_to_scipy(mu, sigma)` to map the distribution parameters to scipy.

    The `scipy.stats.beta()` distribution is defined as:
    $$
    f(x, \\alpha, \\beta) = \\frac{\Gamma(\\alpha + \\beta) x^{\\alpha - 1} {1 - x}^{\\beta - 1}}{\Gamma(\\alpha) \Gamma(\\beta)}
    $$

    with the paramters $\\alpha, \\beta >0$. The parameters can be mapped as follows:
    $$
    \\alpha = -\mu (\sigma^2 + \mu^2 - \mu) / (\sigma^2) \Leftrightarrow \mu = \\alpha / (\\alpha + \\beta)
    $$
    and
    $$
    \\beta = (\sigma^2 + \mu^2 - \mu) (\mu - 1) / (\sigma^2) \Leftrightarrow \sigma = \sqrt{(\\ alpha \\beta) / ( (\\alpha + \\beta)^2 (\\alpha + \\beta + 1) )}
    $$


    Args:
        loc_link (LinkFunction, optional): The link function for $\mu$. Defaults to LogLink().  LOGIT????
        scale_link (LinkFunction, optional): The link function for $\sigma$. Defaults to LogLink().
    """

    def __init__(
        self, loc_link: LinkFunction = LogLink(), scale_link: LinkFunction = LogLink()
    ):
        self.loc_link = loc_link
        self.scale_link = scale_link
        # Set up links as dict
        self.links = {0: self.loc_link, 1: self.scale_link}
        # Set distribution params
        self.n_params = 2
        self.corresponding_gamlss = "BE"
        self.scipy_dist = st.beta

    def theta_to_params(self, theta):
        mu = theta[:, 0]
        sigma = theta[:, 1]
        return mu, sigma

    @staticmethod #####this has to be changed!!!!
    def gamlss_to_scipy(mu: np.ndarray, sigma: np.ndarray):
        """Map GAMLSS Parameters to scipy parameters.

        Args:
            mu (np.ndarray): mu parameter
            sigma (np.ndarray): sigma parameter

        Returns:
            tuple: Tuple of (alpha, loc, scale) for scipy.stats.gamma(alpha, loc, scale)
        """
        alpha = 1 / sigma**2
        beta = 1 / (sigma**2 * mu)
        loc = 0
        scale = 1 / beta
        return alpha, loc, scale

    def dl1_dp1(self, y, theta, param=0):  ###this has to be changed!!!!
        mu, sigma = self.theta_to_params(theta)

        if param == 0:
            return (y - mu) / ((sigma**2) * (mu**2))

        if param == 1:
            return (2 / sigma**3) * (
                (y / mu)
                - np.log(y)
                + np.log(mu)
                + np.log(sigma**2)
                - 1
                + spc.digamma(1 / (sigma**2))
            )

    def dl2_dp2(self, y, theta, param=0):    ###this has to be changed!!!!
        mu, sigma = self.theta_to_params(theta)
        if param == 0:
            # MU
            return -1 / ((sigma**2) * (mu**2))

        if param == 1:
            # SIGMA
            return (4 / sigma**4) - (4 / sigma**6) * spc.polygamma(1, (1 / sigma**2))

    def dl2_dpp(self, y, theta, params=(0, 1)):    ###this has to be changed!!!!
        if sorted(params) == [0, 1]:
            return np.zeros_like(y)

    def link_function(self, y, param=0):   ###this has to be changed!!!!
        return self.links[param].link(y)

    def link_inverse(self, y, param=0):    ###this has to be changed!!!!
        return self.links[param].inverse(y)

    def link_function_derivative(self, y: np.ndarray, param: int = 0) -> np.ndarray:    ###this has to be changed!!!!
        return self.links[param].link_derivative(y)

    def link_inverse_derivative(self, y: np.ndarray, param: int = 0) -> np.ndarray:    ###this has to be changed!!!!
        return self.links[param].inverse_derivative(y)

    def initial_values(self, y, param=0, axis=None):         ###change or leave unchanged
        if param == 0:
            return (y + np.mean(y, axis=None)) / 2
        if param == 1:
            return np.ones_like(y)

    def cdf(self, y, theta):
        mu, sigma = self.theta_to_params(theta)
        return self.scipy_dist(*self.gamlss_to_scipy(mu, sigma)).cdf(y)

    def pdf(self, y, theta):
        mu, sigma = self.theta_to_params(theta)
        return self.scipy_dist(*self.gamlss_to_scipy(mu, sigma)).pdf(y)

    def ppf(self, q, theta):
        mu, sigma = self.theta_to_params(theta)
        return self.scipy_dist(*self.gamlss_to_scipy(mu, sigma)).ppf(q)

    def rvs(self, size, theta):
        mu, sigma = self.theta_to_params(theta)
        return (
            self.scipy_dist(*self.gamlss_to_scipy(mu, sigma))
            .rvs((size, theta.shape[0]))
            .T
        )
