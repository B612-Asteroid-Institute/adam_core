from ...utils.helpers.orbits import make_real_orbits


def test_benchmark_iterate_real_orbits(benchmark):
    orbits = make_real_orbits(27)

    def noop_iterate(iterator):
        for _ in iterator:
            pass

    benchmark(noop_iterate, orbits)


def test_benchmark_iterate_real_orbits_dataframe(benchmark):
    orbits = make_real_orbits(27).to_dataframe().itterrows()

    def noop_iterate(iterator):
        for _ in iterator:
            pass

    benchmark(noop_iterate, orbits)
