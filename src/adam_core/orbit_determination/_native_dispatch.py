"""Keep native specifications paired with the Python behavior they describe."""

import warnings


def _has_compatible_native_spec(
    instance: object, spec_name: str, method_names: tuple[str, ...]
) -> bool:
    """Reject Python method overrides below the class supplying a native spec.

    Defining a specification on a class explicitly declares that it represents
    that class's Python methods, including inherited methods. A subclass that
    changes one of those methods must supply a new specification. Instance
    method replacements conservatively use the Python composition.
    """
    if any(name in getattr(instance, "__dict__", {}) for name in method_names):
        return False
    for cls in type(instance).__mro__:
        if spec_name in cls.__dict__:
            for name in method_names:
                implementation = getattr(cls, name, None)
                if implementation is None or (
                    getattr(getattr(instance, name), "__func__", None)
                    is not implementation
                ):
                    return False
            return True
    return False


#: Methods whose override below a `NativeBackend` declaration voids it: the
#: dynamics the compiled work units reproduce.
_PROPAGATION_METHODS = (
    "propagate_orbits",
    "_propagate_orbits",
    "generate_ephemeris",
    "_generate_ephemeris",
)


def native_backend(propagator: object):
    """
    The `NativeBackend` declared for ``propagator``, or None.

    The declaration is the one on the nearest class in the MRO that sets
    ``native_backend``. It holds only while every propagation method and
    declared work unit the instance uses is the declaring class's own (or
    inherited by it): a subclass or instance that overrides one of them is
    not covered, so OD runs the composed path on its Python behavior.
    """
    for cls in type(propagator).__mro__:
        if "native_backend" not in cls.__dict__:
            continue
        backend = cls.__dict__["native_backend"]
        if backend is None:
            return None
        names = _PROPAGATION_METHODS + tuple(sorted(backend.work_units))
        if any(name in getattr(propagator, "__dict__", {}) for name in names):
            return None
        for name in names:
            declared = getattr(cls, name, None)
            if declared is None:
                continue
            bound = getattr(propagator, name, None)
            if getattr(bound, "__func__", bound) is not declared:
                return None
        return backend
    return None


def native_work_unit(propagator: object, name: str):
    """The bound work unit ``name`` when ``propagator``'s `native_backend`
    declares it (see `native_backend`), else None."""
    backend = native_backend(propagator)
    if backend is None or name not in backend.work_units:
        return None
    return getattr(propagator, name, None)


def _warn_if_analytic_jacobian_mismatches(propagator: object, jacobian: str) -> None:
    """
    Make the analytic two-body Jacobian audible on other dynamics.

    ``jacobian="analytic"`` is the exact derivative of the Kepler two-body
    model only. A propagator declares that it integrates exactly that model
    with ``Propagator.two_body_dynamics = True``; with any other propagator
    the solver still steps along the two-body derivative and only the
    covariance probe checks the result, which this warning says out loud.
    """
    if jacobian != "analytic" or getattr(propagator, "two_body_dynamics", False):
        return
    warnings.warn(
        "jacobian='analytic' is the Kepler two-body Jacobian, but "
        f"{type(propagator).__name__} does not declare two-body dynamics "
        "(Propagator.two_body_dynamics): the solver steps along the two-body "
        "derivative and only the covariance probe checks the result. Pass "
        "jacobian='central' to difference the propagator itself.",
        UserWarning,
        stacklevel=3,
    )
