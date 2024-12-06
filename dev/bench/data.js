window.BENCHMARK_DATA = {
  "lastUpdate": 1733451497877,
  "repoUrl": "https://github.com/B612-Asteroid-Institute/adam_core",
  "entries": {
    "Python Benchmark": [
      {
        "commit": {
          "author": {
            "email": "akoumjian@gmail.com",
            "name": "Alec Koumjian",
            "username": "akoumjian"
          },
          "committer": {
            "email": "akoumjian@gmail.com",
            "name": "Alec Koumjian",
            "username": "akoumjian"
          },
          "distinct": true,
          "id": "afe1e7a2f23d24d696c8afd1d56ae4eb97ea0ccb",
          "message": "Add benchmarking",
          "timestamp": "2024-12-05T21:06:44-05:00",
          "tree_id": "d08b9d11b39aa74880ec6203f2ac8a4ebe686be7",
          "url": "https://github.com/B612-Asteroid-Institute/adam_core/commit/afe1e7a2f23d24d696c8afd1d56ae4eb97ea0ccb"
        },
        "date": 1733451486254,
        "tool": "pytest",
        "benches": [
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 456.9404455808182,
            "unit": "iter/sec",
            "range": "stddev: 0.00012468572407132918",
            "extra": "mean: 2.188469000000421 msec\nrounds: 9"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 321.4069705840881,
            "unit": "iter/sec",
            "range": "stddev: 0.0001874781545414973",
            "extra": "mean: 3.111320200002865 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 317.9933348585281,
            "unit": "iter/sec",
            "range": "stddev: 0.00018223691519830902",
            "extra": "mean: 3.1447200000116027 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 468.307222608728,
            "unit": "iter/sec",
            "range": "stddev: 0.00004434049025183067",
            "extra": "mean: 2.135350367712571 msec\nrounds: 446"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 323.3775962331245,
            "unit": "iter/sec",
            "range": "stddev: 0.00037140360059396147",
            "extra": "mean: 3.092360174757113 msec\nrounds: 309"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 328.5009311886662,
            "unit": "iter/sec",
            "range": "stddev: 0.000051945229692030585",
            "extra": "mean: 3.0441314013374146 msec\nrounds: 299"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_to_matrix",
            "value": 636.319041892481,
            "unit": "iter/sec",
            "range": "stddev: 0.000018816518421798092",
            "extra": "mean: 1.5715387001870837 msec\nrounds: 537"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_from_matrix",
            "value": 10931.631999647476,
            "unit": "iter/sec",
            "range": "stddev: 0.000007389895011691361",
            "extra": "mean: 91.47764945181544 usec\nrounds: 2556"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body",
            "value": 45910.169105346184,
            "unit": "iter/sec",
            "range": "stddev: 0.000008791586148022468",
            "extra": "mean: 21.781666665295536 usec\nrounds: 9"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body_vmap",
            "value": 35550.13840265017,
            "unit": "iter/sec",
            "range": "stddev: 0.00001007162875101048",
            "extra": "mean: 28.129285705550238 usec\nrounds: 7"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body",
            "value": 358.3621331061171,
            "unit": "iter/sec",
            "range": "stddev: 0.0001040045061284084",
            "extra": "mean: 2.7904734000003373 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body_matrix",
            "value": 4.490412441365133,
            "unit": "iter/sec",
            "range": "stddev: 0.0011613951052312174",
            "extra": "mean: 222.69669280000244 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=1,]",
            "value": 571.8400463810686,
            "unit": "iter/sec",
            "range": "stddev: 0.000036842372604879246",
            "extra": "mean: 1.7487407647095945 msec\nrounds: 102"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.698736064056259,
            "unit": "iter/sec",
            "range": "stddev: 0.0006565712249156763",
            "extra": "mean: 212.82319040000175 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=1,]",
            "value": 1150.0979107105518,
            "unit": "iter/sec",
            "range": "stddev: 0.00004400438868154611",
            "extra": "mean: 869.4911891303076 usec\nrounds: 920"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=10000,]",
            "value": 16.655577040565017,
            "unit": "iter/sec",
            "range": "stddev: 0.00036556599699377035",
            "extra": "mean: 60.039949235290884 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 571.1152149683387,
            "unit": "iter/sec",
            "range": "stddev: 0.00004274465067453681",
            "extra": "mean: 1.7509601806974058 msec\nrounds: 487"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.681811886781459,
            "unit": "iter/sec",
            "range": "stddev: 0.0024501042413167078",
            "extra": "mean: 213.59252020000667 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1164.8306100046193,
            "unit": "iter/sec",
            "range": "stddev: 0.000025722807948772053",
            "extra": "mean: 858.4939229885402 usec\nrounds: 870"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 16.429915641428824,
            "unit": "iter/sec",
            "range": "stddev: 0.0015465088032369481",
            "extra": "mean: 60.86458517647235 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=1,]",
            "value": 575.5982428008373,
            "unit": "iter/sec",
            "range": "stddev: 0.00004448239044543616",
            "extra": "mean: 1.7373228853758158 msec\nrounds: 506"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.743138907749659,
            "unit": "iter/sec",
            "range": "stddev: 0.00042768006405461073",
            "extra": "mean: 210.83084840001902 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=1,]",
            "value": 1175.2550874604813,
            "unit": "iter/sec",
            "range": "stddev: 0.00003896768233258606",
            "extra": "mean: 850.8791075823576 usec\nrounds: 976"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=10000,]",
            "value": 17.030763222053924,
            "unit": "iter/sec",
            "range": "stddev: 0.00046696875462235734",
            "extra": "mean: 58.717274555555655 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 576.3125130792315,
            "unit": "iter/sec",
            "range": "stddev: 0.00003976134697607737",
            "extra": "mean: 1.7351696819092317 msec\nrounds: 503"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.717059359610265,
            "unit": "iter/sec",
            "range": "stddev: 0.0008330725386084342",
            "extra": "mean: 211.99648419998312 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1181.4201962230873,
            "unit": "iter/sec",
            "range": "stddev: 0.00002145539820629562",
            "extra": "mean: 846.4388904108173 usec\nrounds: 949"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 16.87659387329921,
            "unit": "iter/sec",
            "range": "stddev: 0.00044041266729756264",
            "extra": "mean: 59.253662647065276 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/orbits/tests/test_benchmarks.py::test_benchmark_iterate_real_orbits",
            "value": 19185.554456125898,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019057649987112766",
            "extra": "mean: 52.12254888368383 usec\nrounds: 14422"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1414.481679540648,
            "unit": "iter/sec",
            "range": "stddev: 0.000040573971037272535",
            "extra": "mean: 706.9727480137808 usec\nrounds: 881"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 16.91877908313838,
            "unit": "iter/sec",
            "range": "stddev: 0.0002685613894828253",
            "extra": "mean: 59.10591982353039 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1436.2462672565139,
            "unit": "iter/sec",
            "range": "stddev: 0.000021317250074662973",
            "extra": "mean: 696.2594248618506 usec\nrounds: 1271"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 18.51421674206519,
            "unit": "iter/sec",
            "range": "stddev: 0.0003291181042875117",
            "extra": "mean: 54.01254689473047 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1420.930469612284,
            "unit": "iter/sec",
            "range": "stddev: 0.00003192946479238796",
            "extra": "mean: 703.7642033764402 usec\nrounds: 1244"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 17.734512835421878,
            "unit": "iter/sec",
            "range": "stddev: 0.00028552513946574073",
            "extra": "mean: 56.387226944439014 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1406.1679806043765,
            "unit": "iter/sec",
            "range": "stddev: 0.00005635923821357754",
            "extra": "mean: 711.152589017278 usec\nrounds: 1202"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 16.707189890229184,
            "unit": "iter/sec",
            "range": "stddev: 0.00036438793182264474",
            "extra": "mean: 59.854470235286364 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1447.0345665502316,
            "unit": "iter/sec",
            "range": "stddev: 0.000028106358461324258",
            "extra": "mean: 691.0684949178693 usec\nrounds: 1279"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 18.51720726201405,
            "unit": "iter/sec",
            "range": "stddev: 0.00041887390555973196",
            "extra": "mean: 54.003823894728804 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1425.8798917134427,
            "unit": "iter/sec",
            "range": "stddev: 0.000026373718556609582",
            "extra": "mean: 701.3213425699735 usec\nrounds: 1191"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 17.520275360559587,
            "unit": "iter/sec",
            "range": "stddev: 0.0002969347066717091",
            "extra": "mean: 57.07672850000575 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1421.9952431659121,
            "unit": "iter/sec",
            "range": "stddev: 0.000024952472876191923",
            "extra": "mean: 703.2372328993258 usec\nrounds: 1228"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 17.036570812697796,
            "unit": "iter/sec",
            "range": "stddev: 0.0003931829863574667",
            "extra": "mean: 58.697258444444365 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1438.9076871578661,
            "unit": "iter/sec",
            "range": "stddev: 0.000025502879885350233",
            "extra": "mean: 694.9716155698649 usec\nrounds: 1246"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 17.640535301691006,
            "unit": "iter/sec",
            "range": "stddev: 0.0007865812813844411",
            "extra": "mean: 56.68762216666639 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1433.7085474941427,
            "unit": "iter/sec",
            "range": "stddev: 0.00006661114942747148",
            "extra": "mean: 697.4918310613513 usec\nrounds: 1320"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 18.57781733614352,
            "unit": "iter/sec",
            "range": "stddev: 0.00033229408877551397",
            "extra": "mean: 53.827636578947285 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1432.421299660395,
            "unit": "iter/sec",
            "range": "stddev: 0.000020408954669312273",
            "extra": "mean: 698.1186332799467 usec\nrounds: 1238"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 16.904983091310452,
            "unit": "iter/sec",
            "range": "stddev: 0.0007926861258845588",
            "extra": "mean: 59.15415558824326 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1427.074601483798,
            "unit": "iter/sec",
            "range": "stddev: 0.000023983838258096536",
            "extra": "mean: 700.7342145675159 usec\nrounds: 1263"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 17.431266528429745,
            "unit": "iter/sec",
            "range": "stddev: 0.0012978938677041256",
            "extra": "mean: 57.368177944444675 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1445.5959931930795,
            "unit": "iter/sec",
            "range": "stddev: 0.000021170915739978813",
            "extra": "mean: 691.756206235165 usec\nrounds: 1251"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 18.48577264788194,
            "unit": "iter/sec",
            "range": "stddev: 0.001186495584218004",
            "extra": "mean: 54.095656105268496 msec\nrounds: 19"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "akoumjian@gmail.com",
            "name": "Alec Koumjian",
            "username": "akoumjian"
          },
          "committer": {
            "email": "akoumjian@gmail.com",
            "name": "Alec Koumjian",
            "username": "akoumjian"
          },
          "distinct": true,
          "id": "afe1e7a2f23d24d696c8afd1d56ae4eb97ea0ccb",
          "message": "Add benchmarking",
          "timestamp": "2024-12-05T21:06:44-05:00",
          "tree_id": "d08b9d11b39aa74880ec6203f2ac8a4ebe686be7",
          "url": "https://github.com/B612-Asteroid-Institute/adam_core/commit/afe1e7a2f23d24d696c8afd1d56ae4eb97ea0ccb"
        },
        "date": 1733451497127,
        "tool": "pytest",
        "benches": [
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 434.8605432663928,
            "unit": "iter/sec",
            "range": "stddev: 0.00016011228000812358",
            "extra": "mean: 2.29958780000743 msec\nrounds: 10"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 301.13151370916387,
            "unit": "iter/sec",
            "range": "stddev: 0.00025411825053401184",
            "extra": "mean: 3.3208081999873684 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 302.88858788507343,
            "unit": "iter/sec",
            "range": "stddev: 0.00022985579485854956",
            "extra": "mean: 3.3015439999985574 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 463.87880516775635,
            "unit": "iter/sec",
            "range": "stddev: 0.00002513402242435292",
            "extra": "mean: 2.1557354827590403 msec\nrounds: 435"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 329.0598325639223,
            "unit": "iter/sec",
            "range": "stddev: 0.00003150289371573855",
            "extra": "mean: 3.038961006599742 msec\nrounds: 303"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 323.2019640532046,
            "unit": "iter/sec",
            "range": "stddev: 0.00002825196107415803",
            "extra": "mean: 3.0940406037736294 msec\nrounds: 318"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_to_matrix",
            "value": 604.6328194566905,
            "unit": "iter/sec",
            "range": "stddev: 0.000017237703217849194",
            "extra": "mean: 1.6538963281857202 msec\nrounds: 518"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_from_matrix",
            "value": 10716.27214439174,
            "unit": "iter/sec",
            "range": "stddev: 0.000005421146750949318",
            "extra": "mean: 93.31603252753717 usec\nrounds: 2275"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body",
            "value": 42224.82607822684,
            "unit": "iter/sec",
            "range": "stddev: 0.000010367502096696181",
            "extra": "mean: 23.68275000463882 usec\nrounds: 8"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body_vmap",
            "value": 32836.871124197816,
            "unit": "iter/sec",
            "range": "stddev: 0.000012355345406389942",
            "extra": "mean: 30.453571420301678 usec\nrounds: 7"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body",
            "value": 341.39550190906266,
            "unit": "iter/sec",
            "range": "stddev: 0.0001231660243427758",
            "extra": "mean: 2.929154000003109 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body_matrix",
            "value": 4.506316599963296,
            "unit": "iter/sec",
            "range": "stddev: 0.0006383814717562756",
            "extra": "mean: 221.9107285999712 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=1,]",
            "value": 550.3539534593772,
            "unit": "iter/sec",
            "range": "stddev: 0.00008174709579505821",
            "extra": "mean: 1.8170124766330258 msec\nrounds: 107"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.550595821023798,
            "unit": "iter/sec",
            "range": "stddev: 0.0018588575306068243",
            "extra": "mean: 219.7514434000027 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=1,]",
            "value": 1141.3706945843096,
            "unit": "iter/sec",
            "range": "stddev: 0.00002999816530674483",
            "extra": "mean: 876.1395440980748 usec\nrounds: 737"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=10000,]",
            "value": 16.220354748182995,
            "unit": "iter/sec",
            "range": "stddev: 0.0012002593614428877",
            "extra": "mean: 61.650932764711584 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 564.7411337643296,
            "unit": "iter/sec",
            "range": "stddev: 0.000025781745196201708",
            "extra": "mean: 1.7707227970706079 msec\nrounds: 478"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.549092177388056,
            "unit": "iter/sec",
            "range": "stddev: 0.0004039577986147517",
            "extra": "mean: 219.82407939998438 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1144.6158817075004,
            "unit": "iter/sec",
            "range": "stddev: 0.0000334242001566452",
            "extra": "mean: 873.6555345608458 usec\nrounds: 868"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 15.999942957379611,
            "unit": "iter/sec",
            "range": "stddev: 0.0012639975869861127",
            "extra": "mean: 62.500222823530294 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=1,]",
            "value": 563.1029760538721,
            "unit": "iter/sec",
            "range": "stddev: 0.000025881100656249098",
            "extra": "mean: 1.7758741163256255 msec\nrounds: 490"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.5805123000145676,
            "unit": "iter/sec",
            "range": "stddev: 0.00032620677368719396",
            "extra": "mean: 218.31619139999248 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=1,]",
            "value": 1151.1482856511095,
            "unit": "iter/sec",
            "range": "stddev: 0.000028337224333161978",
            "extra": "mean: 868.6978145777132 usec\nrounds: 933"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=10000,]",
            "value": 16.283209955117236,
            "unit": "iter/sec",
            "range": "stddev: 0.0005772765247150858",
            "extra": "mean: 61.412952529408095 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 558.1297767595598,
            "unit": "iter/sec",
            "range": "stddev: 0.000053333937773073346",
            "extra": "mean: 1.7916979914705322 msec\nrounds: 469"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.560439923019517,
            "unit": "iter/sec",
            "range": "stddev: 0.0007511448422170557",
            "extra": "mean: 219.27709100000357 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1161.458085546293,
            "unit": "iter/sec",
            "range": "stddev: 0.000015140263348884672",
            "extra": "mean: 860.9867307692374 usec\nrounds: 910"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 16.366672995883775,
            "unit": "iter/sec",
            "range": "stddev: 0.00018941846671879846",
            "extra": "mean: 61.09977270588227 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/orbits/tests/test_benchmarks.py::test_benchmark_iterate_real_orbits",
            "value": 17151.846876950676,
            "unit": "iter/sec",
            "range": "stddev: 0.0000022149399226049953",
            "extra": "mean: 58.30275929899066 usec\nrounds: 14383"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1389.2840666435345,
            "unit": "iter/sec",
            "range": "stddev: 0.00002195364499699076",
            "extra": "mean: 719.7951981238565 usec\nrounds: 959"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 16.218362382569826,
            "unit": "iter/sec",
            "range": "stddev: 0.000321916056849173",
            "extra": "mean: 61.658506352942176 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1399.6507957118413,
            "unit": "iter/sec",
            "range": "stddev: 0.00003188242822324328",
            "extra": "mean: 714.4639241900442 usec\nrounds: 1174"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 17.91011915951884,
            "unit": "iter/sec",
            "range": "stddev: 0.00019017102553437623",
            "extra": "mean: 55.834357722211 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1395.6197619073375,
            "unit": "iter/sec",
            "range": "stddev: 0.000020014986929839504",
            "extra": "mean: 716.5275437439636 usec\nrounds: 1223"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 17.050426482869554,
            "unit": "iter/sec",
            "range": "stddev: 0.00039158330528511363",
            "extra": "mean: 58.64955935294012 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1399.3434640516539,
            "unit": "iter/sec",
            "range": "stddev: 0.00001980279913168461",
            "extra": "mean: 714.6208387643471 usec\nrounds: 1166"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 16.195517052873736,
            "unit": "iter/sec",
            "range": "stddev: 0.00039433657261567966",
            "extra": "mean: 61.74548158822504 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1408.7426858654646,
            "unit": "iter/sec",
            "range": "stddev: 0.000021117138625878786",
            "extra": "mean: 709.8528425619811 usec\nrounds: 1264"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 17.92778785654185,
            "unit": "iter/sec",
            "range": "stddev: 0.00019462693113162177",
            "extra": "mean: 55.779330277778804 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1406.834347176946,
            "unit": "iter/sec",
            "range": "stddev: 0.00002115462048208787",
            "extra": "mean: 710.8157417443434 usec\nrounds: 1181"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 16.920471230224717,
            "unit": "iter/sec",
            "range": "stddev: 0.00027648306798460276",
            "extra": "mean: 59.100008882360136 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1399.050708702165,
            "unit": "iter/sec",
            "range": "stddev: 0.00003636915810626139",
            "extra": "mean: 714.7703752122424 usec\nrounds: 1178"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 16.537795747382656,
            "unit": "iter/sec",
            "range": "stddev: 0.0004091940913568702",
            "extra": "mean: 60.46755052941468 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1396.8961194424605,
            "unit": "iter/sec",
            "range": "stddev: 0.000017786489478451768",
            "extra": "mean: 715.872845576468 usec\nrounds: 1198"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 16.920771701452292,
            "unit": "iter/sec",
            "range": "stddev: 0.0012545307304510863",
            "extra": "mean: 59.09895941177263 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1412.3751723273535,
            "unit": "iter/sec",
            "range": "stddev: 0.000018285424696350612",
            "extra": "mean: 708.0271726613337 usec\nrounds: 1251"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 17.718392515928546,
            "unit": "iter/sec",
            "range": "stddev: 0.0012679973373724465",
            "extra": "mean: 56.438528444440784 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1393.4503668987218,
            "unit": "iter/sec",
            "range": "stddev: 0.00003350697150882394",
            "extra": "mean: 717.6430705785458 usec\nrounds: 1176"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 16.442746841526123,
            "unit": "iter/sec",
            "range": "stddev: 0.00029711671222179477",
            "extra": "mean: 60.81708911764682 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1401.4683496373004,
            "unit": "iter/sec",
            "range": "stddev: 0.000020770824033324482",
            "extra": "mean: 713.537341217017 usec\nrounds: 1184"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 16.844539304301342,
            "unit": "iter/sec",
            "range": "stddev: 0.0003050534766458093",
            "extra": "mean: 59.366420294121355 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1409.075563699455,
            "unit": "iter/sec",
            "range": "stddev: 0.000022366646017675656",
            "extra": "mean: 709.6851480232556 usec\nrounds: 1189"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 17.689876319355108,
            "unit": "iter/sec",
            "range": "stddev: 0.00020628316406758502",
            "extra": "mean: 56.52950772221428 msec\nrounds: 18"
          }
        ]
      }
    ]
  }
}