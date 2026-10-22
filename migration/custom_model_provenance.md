# Migrating custom observation-model provenance for 0.7

`run_od` does not accept live third-party Python observation-model behavior in
reproducible fits. A package name, version, source file, wheel `RECORD`, and
provider-chosen fingerprint cannot bind the class and instance that Python is
actually executing: classes can be created dynamically, module/qualified names
can be borrowed, methods can be replaced on the class or instance, module
delegates can be patched, and editable/duplicate/overlapping installations can
change ownership after metadata inspection.

adam-core deliberately does **not** infer identity from Python source closure,
CPython bytecode, `sys.modules`, import history, mutable closures, absolute
paths, or arbitrary imports. Consequently, `ModelProvenance` and the
`ProvenancedModel` protocol remain data/type declarations only; they do not
make an executable Python `ObservationUncertaintyModel` acceptable to
`run_od`. Subclasses of shipped models, dynamic/notebook/`__main__` classes,
installed classes, and class or instance monkeypatches all fail provenance
preflight before the provenance hook, model application, fitter, or provider
work.

Use an exact shipped adam-core model and express every supported choice through
its constructor configuration or immutable input table. Those classes are
attested by the clean native Core commit plus an import-time exact behavior
graph, and their declarative native specifications execute in Core Rust. For
example:

```python
from adam_core.orbit_determination.observation_uncertainty import IdentityModel

model = IdentityModel()
```

A future external extension mechanism must be declarative: adam-core must own,
validate, and execute the complete behavior in Core Rust. Merely packaging a
Python class more strictly is insufficient. The installable package under
`migration/examples/acme_orbit_models` is retained as a negative migration
fixture proving that even a genuine installed class, and class/instance
monkeypatches of it, are refused.
