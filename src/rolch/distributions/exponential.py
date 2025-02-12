import numpy as np
import scipy.special as spc
import scipy.stats as st

from rolch.base import Distribution, LinkFunction
from rolch.link import LogLink


class DistributionExponential(Distribution):
    """The Exponential Distribution for GAMLSS.

    The distribution function is defined as in GAMLSS as:
    $$
    f(y|\mu)=\\frac{1}{\mu} e^{1/\mu)}
    $$

    with the location parameter $\mu> 0$.

    !!! Note
        The function is parameterized as GAMLSS' EXP() distribution.

        This parameterization is different to the `scipy.stats.expon(alpha, loc, scale)` parameterization.

        We can use `DistributionExponential().gamlss_to_scipy(mu)` to map the distribution parameters to scipy.

    The `scipy.stats.expon()` distribution is defined as:
    $$
    f(x, \\lambda) = \\lambda \exp[-\\lambda x]
    $$

    with the paramters $\\lambda >0$. The parameters can be mapped as follows:
    $$
    \\lambda = 1/mu \Leftrightarrow \mu = 1 / \\lambda
    $$


    Args:
        loc_link (LinkFunction, optional): The link function for $\mu$. Defaults to LogLink().
    """


    def __init__(
        self, loc_link: LinkFunction = LogLink()
    ):
        self.loc_link = loc_link
        # Set up links as dict
        self.links = {0: self.loc_link}
        # Set distribution params
        self.n_params = 1           ###is theta a list of 1 or just a single value???
        self.corresponding_gamlss = "EXP"
        self.scipy_dist = st.expon

    def theta_to_params(self, theta):
        mu = theta[:, 0]                     ###is theta a list of 1 or just a single value???
        return mu
    

    def dl1_dp1(self, y, theta, param=0):
        mu = self.theta_to_params(theta)        ###it's a list right??
        return (-1/mu) + (y/(mu**2))               ###removing the if statement coz it's just 1 par
            
    def dl2_dp2(self, y, theta, param=0):
        mu = self.theta_to_params(theta)
        return (1/(mu**2)) - (2*y / (mu**3))

    ## only one parameter so no cross derivatives

    def link_function(self, y, param=0):
        return self.links[param].link(y)

    def link_inverse(self, y, param=0):
        return self.links[param].inverse(y)

    def link_function_derivative(self, y: np.ndarray, param: int = 0) -> np.ndarray:
        return self.links[param].link_derivative(y)

    def link_inverse_derivative(self, y: np.ndarray, param: int = 0) -> np.ndarray:
        return self.links[param].inverse_derivative(y)
    
    def initial_values(self, y, param=0, axis=None):
        return (y + np.mean(y, axis=None)) / 2  ### this initial value is the same as for gamma
        
    def cdf(self, y, theta):
        mu = self.theta_to_params(theta)
        return self.scipy_dist(*self.gamlss_to_scipy(mu)).cdf(y)

    def pdf(self, y, theta):
        mu = self.theta_to_params(theta)
        return self.scipy_dist(*self.gamlss_to_scipy(mu)).pdf(y)

    def ppf(self, q, theta):
        mu = self.theta_to_params(theta)
        return self.scipy_dist(*self.gamlss_to_scipy(mu)).ppf(q)

    def rvs(self, size, theta):
        mu = self.theta_to_params(theta)
        return (
            self.scipy_dist(*self.gamlss_to_scipy(mu))
            .rvs((size, theta.shape[0]))
            .T
        )











