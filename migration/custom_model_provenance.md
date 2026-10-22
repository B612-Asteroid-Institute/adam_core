# Migrating custom observation-model provenance for 0.7

`run_od` no longer infers a custom model's implementation from Python source,
bytecode, `sys.modules`, import closure, or interpreter metadata. Those signals
cannot simultaneously identify lazily loaded science, executing code after a
disk edit, and equivalent environments on Python 3.11–3.13.

Every model not shipped as an exact adam-core built-in must subclass
`ObservationUncertaintyModel`, implement `model_provenance()`, and return the
frozen public `ModelProvenance` value:

```python
from adam_core.orbit_determination import ModelProvenance
from adam_core.orbit_determination.observation_uncertainty import (
    ObservationUncertaintyModel,
)

class CalibratedModel(ObservationUncertaintyModel):
    def __init__(self, scale):
        self.scale = scale

    def apply(self, observations):
        return observations

    def model_provenance(self):
        return ModelProvenance(
            package="acme-orbit-models",
            version="3.2.0",
            implementation_fingerprint=(
                "ed8ab4b58f6c4b2bbf6c4c4d736942aa"
                "dddc774f5ac86b7b53c50a1072f900a1"
            ),
            configuration={"scale": self.scale},
        )
```

The package/provider chooses the lowercase 64-hex fingerprint. It should be a
release-manifest or content hash covering all science-affecting implementation
and data dependencies, and **must change when that science changes**. adam-core
does not fabricate a weaker fallback. `version` and `package` are mandatory;
`configuration` must include every instance choice and data table/file.

adam-core canonicalizes supported configuration with injective explicit tags,
including NumPy scalars/arrays/types/ufuncs, Arrow tables, distinct
mapping/list/tuple/set/frozenset values, enums, namedtuples, datetimes,
supported public built-in callable partials, nested classes/objects, and file
content. Python function/method values are unsupported without an explicit
portable identity: encode the choice as data and cover its implementation in
the contract fingerprint. Absolute file paths are omitted. Unsupported
or cyclic fields fail closed. A subclass of an adam-core model and an instance
whose `apply` was replaced are custom and need their own contract. A missing or
malformed contract (including an uncontracted `__main__`, dynamic, notebook, or
sourceless model) raises `TypeError` during provenance preflight, before model
application or fitting.

Built-in model descriptors instead use the exact clean adam-core native build
commit, stable class/entrypoint name, and canonical instance configuration/data
hashes. They are independent of import order and Python minor version.


`ProvenancedModel` is a structural protocol for the provenance hook only. It is
useful to type-check providers, but it does **not** replace the runtime
`ObservationUncertaintyModel` subclass requirement. Both are required.
