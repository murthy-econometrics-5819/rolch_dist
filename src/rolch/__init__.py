# ruff: noqa: E402

from importlib.metadata import version
from importlib.util import find_spec

HAS_PANDAS = False
HAS_POLARS = False

if find_spec("pandas") is not None:
    HAS_PANDAS = True

if find_spec("polars") is not None:
    HAS_POLARS = True

from .coordinate_descent import (
    online_coordinate_descent,
    online_coordinate_descent_path,
    soft_threshold,
)
from .distributions import (
    DistributionGamma,
    DistributionJSU,
    DistributionNormal,
    DistributionT,
    DistributionExponential, 
    DistributionBeta, 
    DistributionBetaDebug,
)
from .error import OutOfSupportError
from .estimators import OnlineGamlss, OnlineLasso, OnlineLinearModel
from .gram import (
    init_forget_vector,
    init_gram,
    init_inverted_gram,
    init_y_gram,
    update_gram,
    update_inverted_gram,
    update_y_gram,
)
from .information_criteria import (
    information_criterion,
    select_best_model_by_information_criterion,
)
from .link import (
    IdentityLink,
    LogIdentLink,
    LogLink,
    LogShiftTwoLink,
    LogShiftValueLink,
    SqrtLink,
    SqrtShiftTwoLink,
    SqrtShiftValueLink,
    LogitLink,
)
from .methods import LassoPathMethod, OrdinaryLeastSquaresMethod
from .scaler import OnlineScaler
from .utils import (
    calculate_asymptotic_training_length,
    calculate_effective_training_length,
)
from .warnings import OutOfSupportWarning

__version__ = version("rolch")

__all__ = [
    "OutOfSupportWarning",
    "OutOfSupportError",
    "OnlineScaler",
    "OnlineGamlss",
    "OnlineLinearModel",
    "OnlineLasso",
    "LassoPathMethod",
    "OrdinaryLeastSquaresMethod",
    "IdentityLink",
    "LogLink",
    "LogIdentLink",
    "LogShiftTwoLink",
    "LogShiftValueLink",
    "LogitLink",
    "SqrtLink",
    "SqrtShiftValueLink",
    "SqrtShiftTwoLink",
    "DistributionNormal",
    "DistributionT",
    "DistributionJSU",
    "DistributionGamma",
    "DistributionExponential",
    "DistributionBeta",
    "DistributionBetaDebug",
    "init_forget_vector",
    "init_gram",
    "update_gram",
    "init_inverted_gram",
    "update_inverted_gram",
    "init_y_gram",
    "update_y_gram",
    "online_coordinate_descent",
    "online_coordinate_descent_path",
    "soft_threshold",
    "information_criterion",
    "select_best_model_by_information_criterion",
    "calculate_asymptotic_training_length",
    "calculate_effective_training_length",
]
