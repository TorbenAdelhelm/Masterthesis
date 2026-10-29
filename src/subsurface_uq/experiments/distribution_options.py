"""Shared, explicitly opt-in global diagnostic targets for field CLIs."""
from ..sampling.geostatistical_target import Base10LognormalTarget


def add_distribution_target_arguments(parser):
    parser.add_argument("--distribution-target", action="store_true",
        help="Enable the documented global hydraulic-conductivity diagnostic target; disabled by default")
    parser.add_argument("--target-log10-mean", type=float)
    spread = parser.add_mutually_exclusive_group()
    spread.add_argument("--target-log10-std", type=float,
        help="Explicit global diagnostic standard deviation, separate from residual sigma_R")
    spread.add_argument("--target-log10-variance", type=float)
    parser.add_argument("--target-lb-m-s", type=float)
    parser.add_argument("--target-ub-m-s", type=float)
    parser.add_argument("--minimum-interval-fraction", type=float)


def distribution_target_from_args(args):
    payload = {key: value for key, value in {
        "log10_mean": args.target_log10_mean,
        "log10_std": args.target_log10_std,
        "log10_variance": args.target_log10_variance,
        "interval_m_s": (args.target_lb_m_s if args.target_lb_m_s is not None else 1e-4,
                          args.target_ub_m_s if args.target_ub_m_s is not None else 5e-2)
                         if args.target_lb_m_s is not None or args.target_ub_m_s is not None else None,
        "minimum_interval_fraction": args.minimum_interval_fraction,
    }.items() if value is not None}
    if not args.distribution_target and not payload:
        return None
    return Base10LognormalTarget.from_mapping(payload)
