window.BENCHMARK_DATA = {
  "lastUpdate": 1733452980892,
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
          "id": "33c6481886a0c2bce9dded87330e549054132910",
          "message": "typo",
          "timestamp": "2024-12-05T21:38:58-05:00",
          "tree_id": "a33d9538030fe329876be00e7154709f37b79fe0",
          "url": "https://github.com/B612-Asteroid-Institute/adam_core/commit/33c6481886a0c2bce9dded87330e549054132910"
        },
        "date": 1733452976748,
        "tool": "pytest",
        "benches": [
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 461.1164808410947,
            "unit": "iter/sec",
            "range": "stddev: 0.0001261768627127041",
            "extra": "mean: 2.1686494444439735 msec\nrounds: 9"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 321.15451193289147,
            "unit": "iter/sec",
            "range": "stddev: 0.00021297350575914088",
            "extra": "mean: 3.113765999989937 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 319.7255987031642,
            "unit": "iter/sec",
            "range": "stddev: 0.00018544537335024976",
            "extra": "mean: 3.1276819999902727 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 473.98746026370014,
            "unit": "iter/sec",
            "range": "stddev: 0.00004926975834178913",
            "extra": "mean: 2.109760455358156 msec\nrounds: 448"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 336.61350003287436,
            "unit": "iter/sec",
            "range": "stddev: 0.00009911682757045905",
            "extra": "mean: 2.9707661751603487 msec\nrounds: 314"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 332.5679577629029,
            "unit": "iter/sec",
            "range": "stddev: 0.00003667357780343022",
            "extra": "mean: 3.0069042331279805 msec\nrounds: 326"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 432.7911838305431,
            "unit": "iter/sec",
            "range": "stddev: 0.00007523234071389557",
            "extra": "mean: 2.310583111118881 msec\nrounds: 9"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 273.65273642909744,
            "unit": "iter/sec",
            "range": "stddev: 0.00016913771229805426",
            "extra": "mean: 3.6542663999966862 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 270.94935723528096,
            "unit": "iter/sec",
            "range": "stddev: 0.00014894940423804224",
            "extra": "mean: 3.6907265999957417 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 443.2519438054927,
            "unit": "iter/sec",
            "range": "stddev: 0.000026588752049762133",
            "extra": "mean: 2.256053276190073 msec\nrounds: 420"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 278.63406457849476,
            "unit": "iter/sec",
            "range": "stddev: 0.000036971645114985326",
            "extra": "mean: 3.58893662737453 msec\nrounds: 263"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 273.08331901720794,
            "unit": "iter/sec",
            "range": "stddev: 0.00004408643005698333",
            "extra": "mean: 3.661886063194459 msec\nrounds: 269"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 431.52197238831843,
            "unit": "iter/sec",
            "range": "stddev: 0.00007159327606789399",
            "extra": "mean: 2.3173790999919675 msec\nrounds: 10"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 234.95642169202074,
            "unit": "iter/sec",
            "range": "stddev: 0.00021937910600107233",
            "extra": "mean: 4.256108400011271 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 234.16310434402527,
            "unit": "iter/sec",
            "range": "stddev: 0.0001510996475646877",
            "extra": "mean: 4.270527599987872 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 431.4341153809399,
            "unit": "iter/sec",
            "range": "stddev: 0.00003847642804615494",
            "extra": "mean: 2.3178510098049108 msec\nrounds: 408"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 243.93981257852704,
            "unit": "iter/sec",
            "range": "stddev: 0.00004430312292594874",
            "extra": "mean: 4.099371846807862 msec\nrounds: 235"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 240.1574552533529,
            "unit": "iter/sec",
            "range": "stddev: 0.00003747726191569524",
            "extra": "mean: 4.163934860756478 msec\nrounds: 237"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_to_matrix",
            "value": 638.1329764256698,
            "unit": "iter/sec",
            "range": "stddev: 0.000036950120029337784",
            "extra": "mean: 1.5670714991117227 msec\nrounds: 563"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_from_matrix",
            "value": 11359.013510366245,
            "unit": "iter/sec",
            "range": "stddev: 0.0000055229288822325525",
            "extra": "mean: 88.03581394523381 usec\nrounds: 2854"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body",
            "value": 44317.510314702435,
            "unit": "iter/sec",
            "range": "stddev: 0.000009879035246190177",
            "extra": "mean: 22.5644444577079 usec\nrounds: 9"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body_vmap",
            "value": 35043.45390020825,
            "unit": "iter/sec",
            "range": "stddev: 0.000010065970106321223",
            "extra": "mean: 28.53599998583637 usec\nrounds: 7"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body",
            "value": 358.4322146264135,
            "unit": "iter/sec",
            "range": "stddev: 0.00014459692609112372",
            "extra": "mean: 2.7899277999949845 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body_matrix",
            "value": 4.529124773730844,
            "unit": "iter/sec",
            "range": "stddev: 0.0028970159865499808",
            "extra": "mean: 220.79321059999302 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=1,]",
            "value": 569.511487284003,
            "unit": "iter/sec",
            "range": "stddev: 0.000033369289308598676",
            "extra": "mean: 1.7558908333333085 msec\nrounds: 114"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.608586104318404,
            "unit": "iter/sec",
            "range": "stddev: 0.008987387272916373",
            "extra": "mean: 216.98628980002468 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=1,]",
            "value": 1182.3626922217343,
            "unit": "iter/sec",
            "range": "stddev: 0.00001583897820095887",
            "extra": "mean: 845.764169132347 usec\nrounds: 946"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=10000,]",
            "value": 16.6751790676604,
            "unit": "iter/sec",
            "range": "stddev: 0.00035128225950380117",
            "extra": "mean: 59.96937100000236 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 565.9405916125392,
            "unit": "iter/sec",
            "range": "stddev: 0.000035321906903619405",
            "extra": "mean: 1.7669699166668569 msec\nrounds: 480"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.669597571335568,
            "unit": "iter/sec",
            "range": "stddev: 0.00019057806287910603",
            "extra": "mean: 214.15121640000052 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1176.6815330381169,
            "unit": "iter/sec",
            "range": "stddev: 0.000025679229811399842",
            "extra": "mean: 849.8476197022177 usec\nrounds: 944"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 16.790483676922932,
            "unit": "iter/sec",
            "range": "stddev: 0.00031311474412581866",
            "extra": "mean: 59.55754576471276 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=1,]",
            "value": 575.5220373642786,
            "unit": "iter/sec",
            "range": "stddev: 0.000026104609775841212",
            "extra": "mean: 1.737552926000376 msec\nrounds: 500"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.678793849541233,
            "unit": "iter/sec",
            "range": "stddev: 0.0007709062743982788",
            "extra": "mean: 213.73029720000432 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=1,]",
            "value": 1183.9051274979813,
            "unit": "iter/sec",
            "range": "stddev: 0.000022778958963319016",
            "extra": "mean: 844.6622763712163 usec\nrounds: 948"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=10000,]",
            "value": 17.16434844372592,
            "unit": "iter/sec",
            "range": "stddev: 0.0003488409572579079",
            "extra": "mean: 58.26029477777991 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 573.4172809912407,
            "unit": "iter/sec",
            "range": "stddev: 0.00004013748873370081",
            "extra": "mean: 1.7439306995968888 msec\nrounds: 496"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.695980962162642,
            "unit": "iter/sec",
            "range": "stddev: 0.0009969639348993472",
            "extra": "mean: 212.94805239999732 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1190.090176501005,
            "unit": "iter/sec",
            "range": "stddev: 0.000022699915401758796",
            "extra": "mean: 840.2724598064569 usec\nrounds: 933"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 16.956153810977945,
            "unit": "iter/sec",
            "range": "stddev: 0.00045024112423229105",
            "extra": "mean: 58.97563864704794 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/orbits/tests/test_benchmarks.py::test_benchmark_iterate_real_orbits",
            "value": 19498.29539117696,
            "unit": "iter/sec",
            "range": "stddev: 0.000002422966187921146",
            "extra": "mean: 51.28653453739876 usec\nrounds: 15230"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1381.5892097246194,
            "unit": "iter/sec",
            "range": "stddev: 0.00010528336767267257",
            "extra": "mean: 723.8041473987203 usec\nrounds: 1038"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 16.951475688881445,
            "unit": "iter/sec",
            "range": "stddev: 0.00022997439417895828",
            "extra": "mean: 58.991914235284234 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1460.776100833069,
            "unit": "iter/sec",
            "range": "stddev: 0.000016666387186027173",
            "extra": "mean: 684.5676071984666 usec\nrounds: 1278"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 18.58694737979039,
            "unit": "iter/sec",
            "range": "stddev: 0.001115087876424718",
            "extra": "mean: 53.8011960526289 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1440.2590110681954,
            "unit": "iter/sec",
            "range": "stddev: 0.00003435696510019782",
            "extra": "mean: 694.3195580205613 usec\nrounds: 1172"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 17.642484542404848,
            "unit": "iter/sec",
            "range": "stddev: 0.00047511633573782836",
            "extra": "mean: 56.68135900000001 msec\nrounds: 15"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1428.7676680499726,
            "unit": "iter/sec",
            "range": "stddev: 0.00003581010733881153",
            "extra": "mean: 699.9038558626062 usec\nrounds: 1228"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 16.825440929730824,
            "unit": "iter/sec",
            "range": "stddev: 0.0003178052359250718",
            "extra": "mean: 59.43380647059204 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1452.7877664700818,
            "unit": "iter/sec",
            "range": "stddev: 0.00001636942287295995",
            "extra": "mean: 688.3317873950405 usec\nrounds: 1317"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 18.45292463034544,
            "unit": "iter/sec",
            "range": "stddev: 0.0010754359147067127",
            "extra": "mean: 54.19195168420737 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1443.399811890954,
            "unit": "iter/sec",
            "range": "stddev: 0.0000179743817683352",
            "extra": "mean: 692.8087365412156 usec\nrounds: 1226"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 17.470714942166534,
            "unit": "iter/sec",
            "range": "stddev: 0.0011159273977224436",
            "extra": "mean: 57.23864211111618 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1447.9703789553514,
            "unit": "iter/sec",
            "range": "stddev: 0.000016864753845843536",
            "extra": "mean: 690.6218625283323 usec\nrounds: 1273"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 17.111634879029236,
            "unit": "iter/sec",
            "range": "stddev: 0.0011250373099705004",
            "extra": "mean: 58.43976961111568 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1447.5820132734466,
            "unit": "iter/sec",
            "range": "stddev: 0.00003372319000341245",
            "extra": "mean: 690.8071465593025 usec\nrounds: 1235"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 17.640195213810056,
            "unit": "iter/sec",
            "range": "stddev: 0.0002782870877225826",
            "extra": "mean: 56.68871505555254 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1459.095651151492,
            "unit": "iter/sec",
            "range": "stddev: 0.00003228635614053515",
            "extra": "mean: 685.3560280375164 usec\nrounds: 1284"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 18.58159421094565,
            "unit": "iter/sec",
            "range": "stddev: 0.0007800161926766879",
            "extra": "mean: 53.81669563158048 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1440.5826901797325,
            "unit": "iter/sec",
            "range": "stddev: 0.0000307042423476702",
            "extra": "mean: 694.163553968038 usec\nrounds: 1260"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 16.991051832462404,
            "unit": "iter/sec",
            "range": "stddev: 0.00046732608331687754",
            "extra": "mean: 58.85450823529602 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1444.196175829394,
            "unit": "iter/sec",
            "range": "stddev: 0.00003109929493588885",
            "extra": "mean: 692.4267054132763 usec\nrounds: 1256"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 17.491878202694185,
            "unit": "iter/sec",
            "range": "stddev: 0.00030922290390445194",
            "extra": "mean: 57.16938961111535 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1460.2950917535397,
            "unit": "iter/sec",
            "range": "stddev: 0.00003094488241667004",
            "extra": "mean: 684.7930980848453 usec\nrounds: 1305"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 18.563086662232053,
            "unit": "iter/sec",
            "range": "stddev: 0.00019742418740755817",
            "extra": "mean: 53.870351315795595 msec\nrounds: 19"
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
          "id": "33c6481886a0c2bce9dded87330e549054132910",
          "message": "typo",
          "timestamp": "2024-12-05T21:38:58-05:00",
          "tree_id": "a33d9538030fe329876be00e7154709f37b79fe0",
          "url": "https://github.com/B612-Asteroid-Institute/adam_core/commit/33c6481886a0c2bce9dded87330e549054132910"
        },
        "date": 1733452980621,
        "tool": "pytest",
        "benches": [
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 448.97372769084376,
            "unit": "iter/sec",
            "range": "stddev: 0.00010768252729008168",
            "extra": "mean: 2.2273018181780655 msec\nrounds: 11"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 317.908639288445,
            "unit": "iter/sec",
            "range": "stddev: 0.00016324339517406161",
            "extra": "mean: 3.145557799996368 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 311.7267034665557,
            "unit": "iter/sec",
            "range": "stddev: 0.000157961574644504",
            "extra": "mean: 3.20793819996652 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 460.9773452528832,
            "unit": "iter/sec",
            "range": "stddev: 0.0000280302382699016",
            "extra": "mean: 2.169304002242061 msec\nrounds: 446"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 325.90509599087324,
            "unit": "iter/sec",
            "range": "stddev: 0.00002651767303218152",
            "extra": "mean: 3.068377918300499 msec\nrounds: 306"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 319.61514291027817,
            "unit": "iter/sec",
            "range": "stddev: 0.00008675076307141969",
            "extra": "mean: 3.1287628955700586 msec\nrounds: 316"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 420.79011170218405,
            "unit": "iter/sec",
            "range": "stddev: 0.000056588455977040354",
            "extra": "mean: 2.376481699997157 msec\nrounds: 10"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 264.8027799194795,
            "unit": "iter/sec",
            "range": "stddev: 0.0001529317137323866",
            "extra": "mean: 3.776395400018373 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 258.78998292353276,
            "unit": "iter/sec",
            "range": "stddev: 0.0001316624062172319",
            "extra": "mean: 3.8641371999915464 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 430.4672728382627,
            "unit": "iter/sec",
            "range": "stddev: 0.00005864831970187205",
            "extra": "mean: 2.323056973429255 msec\nrounds: 414"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 265.90373996574397,
            "unit": "iter/sec",
            "range": "stddev: 0.00008542014831502879",
            "extra": "mean: 3.7607594392197288 msec\nrounds: 255"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 265.791128113913,
            "unit": "iter/sec",
            "range": "stddev: 0.00007405533830977831",
            "extra": "mean: 3.762352818531321 msec\nrounds: 259"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 418.4270590416077,
            "unit": "iter/sec",
            "range": "stddev: 0.000060773979630717",
            "extra": "mean: 2.389902800001664 msec\nrounds: 10"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 231.08748801221205,
            "unit": "iter/sec",
            "range": "stddev: 0.00014448936701770814",
            "extra": "mean: 4.32736540001315 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 227.9358860116381,
            "unit": "iter/sec",
            "range": "stddev: 0.00013859590221537891",
            "extra": "mean: 4.38719859999992 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 403.3507967551499,
            "unit": "iter/sec",
            "range": "stddev: 0.00019267477672370976",
            "extra": "mean: 2.479231497854311 msec\nrounds: 233"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 232.03257691608312,
            "unit": "iter/sec",
            "range": "stddev: 0.00014081508835490832",
            "extra": "mean: 4.309739663675156 msec\nrounds: 223"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 228.71608696794226,
            "unit": "iter/sec",
            "range": "stddev: 0.0002098589612407147",
            "extra": "mean: 4.372232899123374 msec\nrounds: 228"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_to_matrix",
            "value": 607.9541274088939,
            "unit": "iter/sec",
            "range": "stddev: 0.00001914895923295299",
            "extra": "mean: 1.6448609441340734 msec\nrounds: 537"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_from_matrix",
            "value": 10833.442722239739,
            "unit": "iter/sec",
            "range": "stddev: 0.000006098802073414717",
            "extra": "mean: 92.30676024594857 usec\nrounds: 2928"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body",
            "value": 44085.231434791414,
            "unit": "iter/sec",
            "range": "stddev: 0.000011799849321268197",
            "extra": "mean: 22.683333339854368 usec\nrounds: 9"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body_vmap",
            "value": 34571.6571958508,
            "unit": "iter/sec",
            "range": "stddev: 0.000010373871263414265",
            "extra": "mean: 28.925428547868897 usec\nrounds: 7"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body",
            "value": 354.8027052144845,
            "unit": "iter/sec",
            "range": "stddev: 0.00011698110081537731",
            "extra": "mean: 2.818467800000235 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body_matrix",
            "value": 4.556218358663876,
            "unit": "iter/sec",
            "range": "stddev: 0.0016754926108807876",
            "extra": "mean: 219.48026219999974 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=1,]",
            "value": 545.9870326034594,
            "unit": "iter/sec",
            "range": "stddev: 0.00003428522147781853",
            "extra": "mean: 1.8315453303563751 msec\nrounds: 112"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.57485534408067,
            "unit": "iter/sec",
            "range": "stddev: 0.0008725935546845609",
            "extra": "mean: 218.58614639999132 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=1,]",
            "value": 1125.3808273878396,
            "unit": "iter/sec",
            "range": "stddev: 0.00001715278074359668",
            "extra": "mean: 888.5880900611525 usec\nrounds: 966"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=10000,]",
            "value": 16.168164055747425,
            "unit": "iter/sec",
            "range": "stddev: 0.0006197434846977861",
            "extra": "mean: 61.849941437507994 msec\nrounds: 16"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 550.302329290012,
            "unit": "iter/sec",
            "range": "stddev: 0.00006602006640656422",
            "extra": "mean: 1.8171829315899461 msec\nrounds: 497"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.559018770195344,
            "unit": "iter/sec",
            "range": "stddev: 0.0004545180366561233",
            "extra": "mean: 219.34544479998976 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1123.8304820345447,
            "unit": "iter/sec",
            "range": "stddev: 0.000019482152764880997",
            "extra": "mean: 889.8139140963981 usec\nrounds: 908"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 16.03060895695621,
            "unit": "iter/sec",
            "range": "stddev: 0.0001390388394307792",
            "extra": "mean: 62.38066206250181 msec\nrounds: 16"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=1,]",
            "value": 555.6906281848661,
            "unit": "iter/sec",
            "range": "stddev: 0.00002289349228814081",
            "extra": "mean: 1.7995624710577662 msec\nrounds: 501"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.557696588929683,
            "unit": "iter/sec",
            "range": "stddev: 0.0020308599163969244",
            "extra": "mean: 219.4090765999931 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=1,]",
            "value": 1135.3448691159183,
            "unit": "iter/sec",
            "range": "stddev: 0.000029771949537802526",
            "extra": "mean: 880.7896412820273 usec\nrounds: 998"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=10000,]",
            "value": 16.44123291115134,
            "unit": "iter/sec",
            "range": "stddev: 0.0001339094510924968",
            "extra": "mean: 60.8226892352912 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 555.5165742935689,
            "unit": "iter/sec",
            "range": "stddev: 0.00006672484374202855",
            "extra": "mean: 1.8001263081514092 msec\nrounds: 503"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.510950815777158,
            "unit": "iter/sec",
            "range": "stddev: 0.00561533211068905",
            "extra": "mean: 221.68275399999402 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1136.5008166125244,
            "unit": "iter/sec",
            "range": "stddev: 0.00003602258035093429",
            "extra": "mean: 879.8937804379399 usec\nrounds: 961"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 16.320742685189092,
            "unit": "iter/sec",
            "range": "stddev: 0.00017396045889890447",
            "extra": "mean: 61.27172147058539 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/orbits/tests/test_benchmarks.py::test_benchmark_iterate_real_orbits",
            "value": 17111.308913204968,
            "unit": "iter/sec",
            "range": "stddev: 0.000001959875262483035",
            "extra": "mean: 58.44088287298057 usec\nrounds: 14369"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1366.9360904779494,
            "unit": "iter/sec",
            "range": "stddev: 0.000023985934942488515",
            "extra": "mean: 731.5630971820707 usec\nrounds: 1029"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 16.25974361190578,
            "unit": "iter/sec",
            "range": "stddev: 0.00022897851461380373",
            "extra": "mean: 61.50158476470538 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1388.8875028813648,
            "unit": "iter/sec",
            "range": "stddev: 0.00002145470157312129",
            "extra": "mean: 720.0007185070176 usec\nrounds: 1286"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 17.79121530258849,
            "unit": "iter/sec",
            "range": "stddev: 0.00030145727299219285",
            "extra": "mean: 56.20751494444044 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1382.4406178913403,
            "unit": "iter/sec",
            "range": "stddev: 0.000031583598375734174",
            "extra": "mean: 723.358375801571 usec\nrounds: 1248"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 17.022915550974833,
            "unit": "iter/sec",
            "range": "stddev: 0.00020678584933981898",
            "extra": "mean: 58.74434358823651 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1381.468730092529,
            "unit": "iter/sec",
            "range": "stddev: 0.00001700928064954726",
            "extra": "mean: 723.8672712722359 usec\nrounds: 1187"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 16.135281572027214,
            "unit": "iter/sec",
            "range": "stddev: 0.00024727649726322576",
            "extra": "mean: 61.975986941166305 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1392.1930305122835,
            "unit": "iter/sec",
            "range": "stddev: 0.00001651141319186247",
            "extra": "mean: 718.291198191124 usec\nrounds: 1216"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 17.740834072164855,
            "unit": "iter/sec",
            "range": "stddev: 0.0011812218784890895",
            "extra": "mean: 56.36713561111466 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1376.2143914561318,
            "unit": "iter/sec",
            "range": "stddev: 0.000018975253016070843",
            "extra": "mean: 726.6309713139459 usec\nrounds: 1255"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 16.795074644795157,
            "unit": "iter/sec",
            "range": "stddev: 0.0011727263199972817",
            "extra": "mean: 59.54126558823619 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1385.1962837244048,
            "unit": "iter/sec",
            "range": "stddev: 0.000015981945103873334",
            "extra": "mean: 721.9193494450333 usec\nrounds: 1262"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 16.413111957040922,
            "unit": "iter/sec",
            "range": "stddev: 0.001156466960726552",
            "extra": "mean: 60.92689811763689 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1388.7627225884094,
            "unit": "iter/sec",
            "range": "stddev: 0.000015855139109197414",
            "extra": "mean: 720.0654105520459 usec\nrounds: 1213"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 16.953169091248004,
            "unit": "iter/sec",
            "range": "stddev: 0.00026016581701237396",
            "extra": "mean: 58.98602170589129 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1391.4577565018853,
            "unit": "iter/sec",
            "range": "stddev: 0.00001736658446653196",
            "extra": "mean: 718.6707575758483 usec\nrounds: 1287"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 17.793629238977292,
            "unit": "iter/sec",
            "range": "stddev: 0.00024258433161370696",
            "extra": "mean: 56.199889666661164 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1380.3603624682935,
            "unit": "iter/sec",
            "range": "stddev: 0.00001743017020416356",
            "extra": "mean: 724.44850431075 usec\nrounds: 1160"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 16.276983568057485,
            "unit": "iter/sec",
            "range": "stddev: 0.00013270522759101079",
            "extra": "mean: 61.43644464705577 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1381.0587918765216,
            "unit": "iter/sec",
            "range": "stddev: 0.000021832055861591595",
            "extra": "mean: 724.0821360264064 usec\nrounds: 1213"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 16.80892383551406,
            "unit": "iter/sec",
            "range": "stddev: 0.00023038159544932108",
            "extra": "mean: 59.492208411771735 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1399.441418634932,
            "unit": "iter/sec",
            "range": "stddev: 0.000016529287983634414",
            "extra": "mean: 714.5708185308948 usec\nrounds: 1306"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 17.768378787997666,
            "unit": "iter/sec",
            "range": "stddev: 0.00016381972673168462",
            "extra": "mean: 56.2797547222197 msec\nrounds: 18"
          }
        ]
      }
    ]
  }
}