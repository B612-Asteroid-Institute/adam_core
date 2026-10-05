"""Keep native specifications paired with the Python behavior they describe."""


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
