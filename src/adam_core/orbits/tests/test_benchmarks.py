from ...utils.helpers.orbits import make_real_orbits


def test_benchmark_iterate_real_orbits(benchmark):
    orbits = make_real_orbits(27)

    def noop_iterate(iterator):
        for _ in iterator:
            pass

    benchmark(noop_iterate, orbits)
