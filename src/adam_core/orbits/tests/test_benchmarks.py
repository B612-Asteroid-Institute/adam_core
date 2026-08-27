from ...utils.helpers.orbits import make_real_orbits


def test_benchmark_group_real_orbits(benchmark):
    orbits = make_real_orbits(27)
    groups = benchmark(lambda: list(orbits.group_by_orbit_id()))
    assert sum(len(group) for _, group in groups) == len(orbits)


def test_benchmark_classify_real_orbits(benchmark):
    orbits = make_real_orbits(27)
    classes = benchmark(orbits.dynamical_class)
    assert len(classes) == len(orbits)
