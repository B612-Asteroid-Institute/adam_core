window.BENCHMARK_DATA = {
  "lastUpdate": 1739500956159,
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
      },
      {
        "commit": {
          "author": {
            "email": "akoumjian@users.noreply.github.com",
            "name": "Alec Koumjian",
            "username": "akoumjian"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "3059a453f938b60e451253742d973635428ac6cf",
          "message": "Use chunking / padding in more jitted functions (#129)\n\n* Use chunking / padding in more jitted functions",
          "timestamp": "2024-12-05T21:50:15-05:00",
          "tree_id": "b5f7b98657f2f2a536f0eeb119da3e52f017ce13",
          "url": "https://github.com/B612-Asteroid-Institute/adam_core/commit/3059a453f938b60e451253742d973635428ac6cf"
        },
        "date": 1733453647494,
        "tool": "pytest",
        "benches": [
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 420.37516381567997,
            "unit": "iter/sec",
            "range": "stddev: 0.00006813199676945155",
            "extra": "mean: 2.3788274999958503 msec\nrounds: 6"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 296.4827070527347,
            "unit": "iter/sec",
            "range": "stddev: 0.00020724688796091163",
            "extra": "mean: 3.3728780000046754 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 321.3583844623688,
            "unit": "iter/sec",
            "range": "stddev: 0.00016485578027437898",
            "extra": "mean: 3.1117905999963114 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 431.81852762262247,
            "unit": "iter/sec",
            "range": "stddev: 0.00007490658640448075",
            "extra": "mean: 2.3157876191776703 msec\nrounds: 365"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 308.6617257191196,
            "unit": "iter/sec",
            "range": "stddev: 0.00008903530084778359",
            "extra": "mean: 3.239792681357566 msec\nrounds: 295"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 333.1396926869508,
            "unit": "iter/sec",
            "range": "stddev: 0.000029126229648258375",
            "extra": "mean: 3.0017437788168144 msec\nrounds: 321"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 430.9261746006565,
            "unit": "iter/sec",
            "range": "stddev: 0.000050349345605534495",
            "extra": "mean: 2.3205831043489287 msec\nrounds: 345"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 277.5373992175199,
            "unit": "iter/sec",
            "range": "stddev: 0.000028539644306118426",
            "extra": "mean: 3.603118004345966 msec\nrounds: 230"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 266.66788978241783,
            "unit": "iter/sec",
            "range": "stddev: 0.0001545312831506169",
            "extra": "mean: 3.7499828000136404 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 433.84791620016193,
            "unit": "iter/sec",
            "range": "stddev: 0.000023358172494887452",
            "extra": "mean: 2.3049551759023217 msec\nrounds: 415"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 277.2065855278268,
            "unit": "iter/sec",
            "range": "stddev: 0.00008419483935412323",
            "extra": "mean: 3.607417904938688 msec\nrounds: 263"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 273.9413086943906,
            "unit": "iter/sec",
            "range": "stddev: 0.00012034878996455575",
            "extra": "mean: 3.6504169625458047 msec\nrounds: 267"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 401.2320208141502,
            "unit": "iter/sec",
            "range": "stddev: 0.00002390970199387018",
            "extra": "mean: 2.4923235138882345 msec\nrounds: 72"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 234.78006385377205,
            "unit": "iter/sec",
            "range": "stddev: 0.00003406689530497829",
            "extra": "mean: 4.2593054264727925 msec\nrounds: 68"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 235.53735846271718,
            "unit": "iter/sec",
            "range": "stddev: 0.00015006111629419435",
            "extra": "mean: 4.245610999998917 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 400.83801826283144,
            "unit": "iter/sec",
            "range": "stddev: 0.00003022132331425527",
            "extra": "mean: 2.4947733359571074 msec\nrounds: 381"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 233.19969258495286,
            "unit": "iter/sec",
            "range": "stddev: 0.00004974974848252404",
            "extra": "mean: 4.288170318387996 msec\nrounds: 223"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 236.97099373870657,
            "unit": "iter/sec",
            "range": "stddev: 0.00013721803183468405",
            "extra": "mean: 4.219925756409828 msec\nrounds: 234"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_to_matrix",
            "value": 652.496556779999,
            "unit": "iter/sec",
            "range": "stddev: 0.0000555072066571038",
            "extra": "mean: 1.532575137154583 msec\nrounds: 576"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_from_matrix",
            "value": 11154.814216996858,
            "unit": "iter/sec",
            "range": "stddev: 0.0000045084470855348856",
            "extra": "mean: 89.64739174914057 usec\nrounds: 3030"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body",
            "value": 44410.23211769373,
            "unit": "iter/sec",
            "range": "stddev: 0.000008798485606555426",
            "extra": "mean: 22.517333333224894 usec\nrounds: 9"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body_vmap",
            "value": 34924.04023048979,
            "unit": "iter/sec",
            "range": "stddev: 0.00000984163397241766",
            "extra": "mean: 28.63357141385287 usec\nrounds: 7"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body",
            "value": 360.3645375572058,
            "unit": "iter/sec",
            "range": "stddev: 0.00007949958144301245",
            "extra": "mean: 2.774967833346409 msec\nrounds: 6"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body_matrix",
            "value": 4.534864626513667,
            "unit": "iter/sec",
            "range": "stddev: 0.0007624261642095966",
            "extra": "mean: 220.513748999997 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=1,]",
            "value": 573.2246944593684,
            "unit": "iter/sec",
            "range": "stddev: 0.00004507129516451562",
            "extra": "mean: 1.7445166086976434 msec\nrounds: 115"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.71769888990553,
            "unit": "iter/sec",
            "range": "stddev: 0.005689187807421065",
            "extra": "mean: 211.9677460000048 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=1,]",
            "value": 1182.6203236533147,
            "unit": "iter/sec",
            "range": "stddev: 0.000033455771593237014",
            "extra": "mean: 845.5799211287275 usec\nrounds: 1027"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=10000,]",
            "value": 16.953757738522267,
            "unit": "iter/sec",
            "range": "stddev: 0.0012257020249116083",
            "extra": "mean: 58.98397366666409 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 566.5890668571486,
            "unit": "iter/sec",
            "range": "stddev: 0.00016791276678346868",
            "extra": "mean: 1.7649475757570967 msec\nrounds: 528"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.758515384694116,
            "unit": "iter/sec",
            "range": "stddev: 0.00015229843436885058",
            "extra": "mean: 210.14957800000502 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1182.6579515450621,
            "unit": "iter/sec",
            "range": "stddev: 0.00003475665920554471",
            "extra": "mean: 845.5530178388165 usec\nrounds: 1009"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 16.930839670160964,
            "unit": "iter/sec",
            "range": "stddev: 0.00015817807524596254",
            "extra": "mean: 59.06381605883419 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=1,]",
            "value": 579.3647469456783,
            "unit": "iter/sec",
            "range": "stddev: 0.00005396203632477719",
            "extra": "mean: 1.7260283875949407 msec\nrounds: 516"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.7907661118270015,
            "unit": "iter/sec",
            "range": "stddev: 0.0002921640894858715",
            "extra": "mean: 208.7348822000081 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=1,]",
            "value": 1192.8650030877916,
            "unit": "iter/sec",
            "range": "stddev: 0.000014589207154331128",
            "extra": "mean: 838.3178292694054 usec\nrounds: 1066"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=10000,]",
            "value": 17.17823607998632,
            "unit": "iter/sec",
            "range": "stddev: 0.0011058181639122774",
            "extra": "mean: 58.21319461111961 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 579.6665751233413,
            "unit": "iter/sec",
            "range": "stddev: 0.000022066546747106267",
            "extra": "mean: 1.7251296571433679 msec\nrounds: 525"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.732263091779223,
            "unit": "iter/sec",
            "range": "stddev: 0.0017519132822426402",
            "extra": "mean: 211.31538560000536 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1185.7324647528308,
            "unit": "iter/sec",
            "range": "stddev: 0.000012812461320144674",
            "extra": "mean: 843.3605638085087 usec\nrounds: 1050"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 17.162277240256902,
            "unit": "iter/sec",
            "range": "stddev: 0.00017353660040619401",
            "extra": "mean: 58.26732583333043 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/orbits/tests/test_benchmarks.py::test_benchmark_iterate_real_orbits",
            "value": 19597.259190851815,
            "unit": "iter/sec",
            "range": "stddev: 0.000001948433733084244",
            "extra": "mean: 51.027543712174264 usec\nrounds: 15545"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1444.1093322031313,
            "unit": "iter/sec",
            "range": "stddev: 0.000016630605381167056",
            "extra": "mean: 692.4683455056698 usec\nrounds: 1068"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 17.09517881937895,
            "unit": "iter/sec",
            "range": "stddev: 0.000288719512137816",
            "extra": "mean: 58.496024555555294 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1458.3986425142818,
            "unit": "iter/sec",
            "range": "stddev: 0.00002839877229457706",
            "extra": "mean: 685.6835784460127 usec\nrounds: 1364"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 18.79983107342358,
            "unit": "iter/sec",
            "range": "stddev: 0.0012347418228000613",
            "extra": "mean: 53.19196731579424 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1443.042438022878,
            "unit": "iter/sec",
            "range": "stddev: 0.000030230529130379882",
            "extra": "mean: 692.980312741257 usec\nrounds: 1295"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 17.881367195863103,
            "unit": "iter/sec",
            "range": "stddev: 0.00019097560974471337",
            "extra": "mean: 55.92413538889534 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1446.3095329026155,
            "unit": "iter/sec",
            "range": "stddev: 0.000029465785379658796",
            "extra": "mean: 691.4149269230691 usec\nrounds: 1300"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 16.955055408193285,
            "unit": "iter/sec",
            "range": "stddev: 0.0011063658827522881",
            "extra": "mean: 58.97945927777767 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1456.8579932080038,
            "unit": "iter/sec",
            "range": "stddev: 0.000028494790105938963",
            "extra": "mean: 686.4086991745834 usec\nrounds: 1333"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 18.868393202700684,
            "unit": "iter/sec",
            "range": "stddev: 0.00010797754480155203",
            "extra": "mean: 52.99868352631464 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1454.9825625337683,
            "unit": "iter/sec",
            "range": "stddev: 0.00002126443095957951",
            "extra": "mean: 687.2934602450201 usec\nrounds: 1308"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 17.69615775006478,
            "unit": "iter/sec",
            "range": "stddev: 0.00047118823309056127",
            "extra": "mean: 56.509441999992305 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1448.8953842618823,
            "unit": "iter/sec",
            "range": "stddev: 0.000026199401537217563",
            "extra": "mean: 690.1809549965781 usec\nrounds: 1311"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 17.047945240567106,
            "unit": "iter/sec",
            "range": "stddev: 0.0024830212904485972",
            "extra": "mean: 58.658095500002595 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1444.8390797953643,
            "unit": "iter/sec",
            "range": "stddev: 0.000027055982919589438",
            "extra": "mean: 692.1185992156525 usec\nrounds: 1275"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 17.848479545141842,
            "unit": "iter/sec",
            "range": "stddev: 0.00009726607139114422",
            "extra": "mean: 56.02718133333598 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1459.4327837820422,
            "unit": "iter/sec",
            "range": "stddev: 0.00002938991805135973",
            "extra": "mean: 685.1977090774632 usec\nrounds: 1344"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 18.803674091842225,
            "unit": "iter/sec",
            "range": "stddev: 0.0010882063315732844",
            "extra": "mean: 53.18109615789605 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1451.4798226155656,
            "unit": "iter/sec",
            "range": "stddev: 0.0000176947299567313",
            "extra": "mean: 688.9520504652974 usec\nrounds: 1288"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 17.16498143072247,
            "unit": "iter/sec",
            "range": "stddev: 0.001114490638928801",
            "extra": "mean: 58.25814633333456 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1407.115766985121,
            "unit": "iter/sec",
            "range": "stddev: 0.00013892296057006547",
            "extra": "mean: 710.6735802858601 usec\nrounds: 1258"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 17.72083482853146,
            "unit": "iter/sec",
            "range": "stddev: 0.0001570372587715769",
            "extra": "mean: 56.430749999991455 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1465.2729771296658,
            "unit": "iter/sec",
            "range": "stddev: 0.00001393277807459175",
            "extra": "mean: 682.4666909226072 usec\nrounds: 1333"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 18.764764981254118,
            "unit": "iter/sec",
            "range": "stddev: 0.001136882966769395",
            "extra": "mean: 53.29136821052615 msec\nrounds: 19"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "akoumjian@users.noreply.github.com",
            "name": "Alec Koumjian",
            "username": "akoumjian"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "3059a453f938b60e451253742d973635428ac6cf",
          "message": "Use chunking / padding in more jitted functions (#129)\n\n* Use chunking / padding in more jitted functions",
          "timestamp": "2024-12-05T21:50:15-05:00",
          "tree_id": "b5f7b98657f2f2a536f0eeb119da3e52f017ce13",
          "url": "https://github.com/B612-Asteroid-Institute/adam_core/commit/3059a453f938b60e451253742d973635428ac6cf"
        },
        "date": 1733453666679,
        "tool": "pytest",
        "benches": [
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 418.6921898244301,
            "unit": "iter/sec",
            "range": "stddev: 0.00008739189784291805",
            "extra": "mean: 2.388389428566435 msec\nrounds: 7"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 291.20513322493275,
            "unit": "iter/sec",
            "range": "stddev: 0.00020087424008978532",
            "extra": "mean: 3.4340053999926567 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 311.9965545573809,
            "unit": "iter/sec",
            "range": "stddev: 0.00017769935105622602",
            "extra": "mean: 3.2051636000232975 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 426.1002936572385,
            "unit": "iter/sec",
            "range": "stddev: 0.0000687654057604987",
            "extra": "mean: 2.3468653152452768 msec\nrounds: 387"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 301.7335318550423,
            "unit": "iter/sec",
            "range": "stddev: 0.000040384942587603976",
            "extra": "mean: 3.31418253003586 msec\nrounds: 283"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 325.14439910943963,
            "unit": "iter/sec",
            "range": "stddev: 0.0001112253449694899",
            "extra": "mean: 3.075556591898765 msec\nrounds: 321"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 436.0552684851778,
            "unit": "iter/sec",
            "range": "stddev: 0.000059936242060991074",
            "extra": "mean: 2.2932872786376883 msec\nrounds: 323"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 275.3519762355028,
            "unit": "iter/sec",
            "range": "stddev: 0.00005581761653484777",
            "extra": "mean: 3.631715354549411 msec\nrounds: 220"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 266.1142843750231,
            "unit": "iter/sec",
            "range": "stddev: 0.00017134735815670576",
            "extra": "mean: 3.757784000015363 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 436.102470565573,
            "unit": "iter/sec",
            "range": "stddev: 0.000024398987991706072",
            "extra": "mean: 2.2930390619046914 msec\nrounds: 420"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 273.2511490020647,
            "unit": "iter/sec",
            "range": "stddev: 0.00013266068226143617",
            "extra": "mean: 3.6596369444449945 msec\nrounds: 252"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 269.9891698790327,
            "unit": "iter/sec",
            "range": "stddev: 0.0001851984780615923",
            "extra": "mean: 3.703852270993111 msec\nrounds: 262"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 393.4568082822837,
            "unit": "iter/sec",
            "range": "stddev: 0.00007314965781401151",
            "extra": "mean: 2.541575031744157 msec\nrounds: 63"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 227.72923870097733,
            "unit": "iter/sec",
            "range": "stddev: 0.00013499196395835736",
            "extra": "mean: 4.391179655736092 msec\nrounds: 61"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 230.60588375269424,
            "unit": "iter/sec",
            "range": "stddev: 0.00015929341771073495",
            "extra": "mean: 4.336402799992811 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 397.1432786851497,
            "unit": "iter/sec",
            "range": "stddev: 0.00003416146114700231",
            "extra": "mean: 2.517982938829459 msec\nrounds: 376"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 230.90893346179604,
            "unit": "iter/sec",
            "range": "stddev: 0.00006175706799095285",
            "extra": "mean: 4.330711614349258 msec\nrounds: 223"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 232.3976489707516,
            "unit": "iter/sec",
            "range": "stddev: 0.00009209647207542742",
            "extra": "mean: 4.3029695198244235 msec\nrounds: 227"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_to_matrix",
            "value": 605.2092202568091,
            "unit": "iter/sec",
            "range": "stddev: 0.00003679751714259128",
            "extra": "mean: 1.6523211585832565 msec\nrounds: 536"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_from_matrix",
            "value": 11258.061181208159,
            "unit": "iter/sec",
            "range": "stddev: 0.0000055706299278173244",
            "extra": "mean: 88.82524121197616 usec\nrounds: 4125"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body",
            "value": 43885.31306059457,
            "unit": "iter/sec",
            "range": "stddev: 0.000010184965311767198",
            "extra": "mean: 22.786666660421258 usec\nrounds: 9"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body_vmap",
            "value": 34825.17760517347,
            "unit": "iter/sec",
            "range": "stddev: 0.000011334144306820752",
            "extra": "mean: 28.714857145522338 usec\nrounds: 7"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body",
            "value": 358.28648200159785,
            "unit": "iter/sec",
            "range": "stddev: 0.00006289746497772453",
            "extra": "mean: 2.79106259999935 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body_matrix",
            "value": 4.549330554408056,
            "unit": "iter/sec",
            "range": "stddev: 0.000463196638701495",
            "extra": "mean: 219.81256100000337 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=1,]",
            "value": 562.2980905926892,
            "unit": "iter/sec",
            "range": "stddev: 0.00003548719930788537",
            "extra": "mean: 1.7784161403535124 msec\nrounds: 114"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.549900426884982,
            "unit": "iter/sec",
            "range": "stddev: 0.0006055317613621965",
            "extra": "mean: 219.78502960000696 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=1,]",
            "value": 1157.3596454536196,
            "unit": "iter/sec",
            "range": "stddev: 0.00003806397491228854",
            "extra": "mean: 864.0356555788297 usec\nrounds: 932"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=10000,]",
            "value": 16.25978221136992,
            "unit": "iter/sec",
            "range": "stddev: 0.0002168199360029027",
            "extra": "mean: 61.50143876470458 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 568.1977335160781,
            "unit": "iter/sec",
            "range": "stddev: 0.00006674315306283244",
            "extra": "mean: 1.759950702041481 msec\nrounds: 490"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.547476738080647,
            "unit": "iter/sec",
            "range": "stddev: 0.0008689308234228491",
            "extra": "mean: 219.90216940000664 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1180.0753588758093,
            "unit": "iter/sec",
            "range": "stddev: 0.000018786589350049065",
            "extra": "mean: 847.4035090035632 usec\nrounds: 833"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 16.087099934740436,
            "unit": "iter/sec",
            "range": "stddev: 0.001312803831424353",
            "extra": "mean: 62.16160799999002 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=1,]",
            "value": 576.0517287868533,
            "unit": "iter/sec",
            "range": "stddev: 0.000023548856745055483",
            "extra": "mean: 1.735955210317602 msec\nrounds: 504"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.572885100659771,
            "unit": "iter/sec",
            "range": "stddev: 0.0004423140686599788",
            "extra": "mean: 218.6803250000139 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=1,]",
            "value": 1176.2875417790062,
            "unit": "iter/sec",
            "range": "stddev: 0.000038038482795558134",
            "extra": "mean: 850.1322716447455 usec\nrounds: 924"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=10000,]",
            "value": 16.554536574384006,
            "unit": "iter/sec",
            "range": "stddev: 0.00023919850858315856",
            "extra": "mean: 60.40640252940514 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 578.9328866071403,
            "unit": "iter/sec",
            "range": "stddev: 0.000022034717212413167",
            "extra": "mean: 1.7273159344264248 msec\nrounds: 488"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.558189974923682,
            "unit": "iter/sec",
            "range": "stddev: 0.0003459699337190932",
            "extra": "mean: 219.38532739999346 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1179.1778194051612,
            "unit": "iter/sec",
            "range": "stddev: 0.000017910920096353264",
            "extra": "mean: 848.0485161300371 usec\nrounds: 930"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 16.54124238813677,
            "unit": "iter/sec",
            "range": "stddev: 0.00022703830690232649",
            "extra": "mean: 60.45495111764948 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/orbits/tests/test_benchmarks.py::test_benchmark_iterate_real_orbits",
            "value": 17164.40053290555,
            "unit": "iter/sec",
            "range": "stddev: 0.0000022027536287106893",
            "extra": "mean: 58.26011797399617 usec\nrounds: 14393"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1403.3279185550825,
            "unit": "iter/sec",
            "range": "stddev: 0.00003621572947655216",
            "extra": "mean: 712.5918231781752 usec\nrounds: 837"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 16.361852996906006,
            "unit": "iter/sec",
            "range": "stddev: 0.00027836503384357723",
            "extra": "mean: 61.11777194117916 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1448.1974655294914,
            "unit": "iter/sec",
            "range": "stddev: 0.000014557299528468986",
            "extra": "mean: 690.5135686274517 usec\nrounds: 1224"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 17.786845625863982,
            "unit": "iter/sec",
            "range": "stddev: 0.0012263276884846605",
            "extra": "mean: 56.22132338889211 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1432.1692563274432,
            "unit": "iter/sec",
            "range": "stddev: 0.000016691149054462458",
            "extra": "mean: 698.2414931628483 usec\nrounds: 1243"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 16.911621153135936,
            "unit": "iter/sec",
            "range": "stddev: 0.0011605487502147581",
            "extra": "mean: 59.130936705885766 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1433.051505223808,
            "unit": "iter/sec",
            "range": "stddev: 0.000017597632413628936",
            "extra": "mean: 697.8116253008117 usec\nrounds: 1241"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 16.153555086519116,
            "unit": "iter/sec",
            "range": "stddev: 0.0010715804755382823",
            "extra": "mean: 61.90587735293922 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1449.0810159114696,
            "unit": "iter/sec",
            "range": "stddev: 0.000015976354556152207",
            "extra": "mean: 690.092540734171 usec\nrounds: 1252"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 17.87666268307206,
            "unit": "iter/sec",
            "range": "stddev: 0.00021455063747029924",
            "extra": "mean: 55.938852666662974 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1441.507335527878,
            "unit": "iter/sec",
            "range": "stddev: 0.000026156934502863193",
            "extra": "mean: 693.7182873466276 usec\nrounds: 1225"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 16.870636515925394,
            "unit": "iter/sec",
            "range": "stddev: 0.00020880145775946775",
            "extra": "mean: 59.274586294122855 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1434.8504324288804,
            "unit": "iter/sec",
            "range": "stddev: 0.00003055677298932575",
            "extra": "mean: 696.9367520120018 usec\nrounds: 1242"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 16.578413627162714,
            "unit": "iter/sec",
            "range": "stddev: 0.00023109292642337024",
            "extra": "mean: 60.3194022352996 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1437.4854576778175,
            "unit": "iter/sec",
            "range": "stddev: 0.000030153203077134316",
            "extra": "mean: 695.659211478527 usec\nrounds: 1272"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 16.961955233394896,
            "unit": "iter/sec",
            "range": "stddev: 0.0004028347575763006",
            "extra": "mean: 58.95546747058902 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1434.3837081230727,
            "unit": "iter/sec",
            "range": "stddev: 0.00003510557223211171",
            "extra": "mean: 697.1635234957633 usec\nrounds: 1213"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 17.529458532615063,
            "unit": "iter/sec",
            "range": "stddev: 0.0035141061833819597",
            "extra": "mean: 57.046827666662615 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1429.5982843046,
            "unit": "iter/sec",
            "range": "stddev: 0.000019374916078031664",
            "extra": "mean: 699.4972021013795 usec\nrounds: 1237"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 16.41659497692864,
            "unit": "iter/sec",
            "range": "stddev: 0.0004214093148302784",
            "extra": "mean: 60.91397158822327 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1428.2431501134183,
            "unit": "iter/sec",
            "range": "stddev: 0.00002589373247286279",
            "extra": "mean: 700.160893416915 usec\nrounds: 1276"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 16.875009002069774,
            "unit": "iter/sec",
            "range": "stddev: 0.0009926948940524665",
            "extra": "mean: 59.259227647069515 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1423.329595362515,
            "unit": "iter/sec",
            "range": "stddev: 0.0000610412584135912",
            "extra": "mean: 702.5779575287374 usec\nrounds: 1295"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 17.749505716449477,
            "unit": "iter/sec",
            "range": "stddev: 0.0011212048172953824",
            "extra": "mean: 56.33959705555311 msec\nrounds: 18"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "akoumjian@users.noreply.github.com",
            "name": "Alec Koumjian",
            "username": "akoumjian"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "3d2a90dabb88ef0a7a9a73d8e76e1fb7f1f56c1a",
          "message": "Ak/more perf improvements (#130)\n\n* this seems like a better chunk size\r\n\r\n* Change chunking",
          "timestamp": "2024-12-09T10:07:50-05:00",
          "tree_id": "4e1a0b418c53f0b92581c519c6d0805ee4c851c1",
          "url": "https://github.com/B612-Asteroid-Institute/adam_core/commit/3d2a90dabb88ef0a7a9a73d8e76e1fb7f1f56c1a"
        },
        "date": 1733757137201,
        "tool": "pytest",
        "benches": [
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 427.71725684441463,
            "unit": "iter/sec",
            "range": "stddev: 0.00009551771899560454",
            "extra": "mean: 2.337993111098058 msec\nrounds: 9"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 293.6016852728394,
            "unit": "iter/sec",
            "range": "stddev: 0.0002124097466365987",
            "extra": "mean: 3.4059750000096756 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 313.8401957120576,
            "unit": "iter/sec",
            "range": "stddev: 0.00021812079999669628",
            "extra": "mean: 3.186334999986684 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 436.65855382456783,
            "unit": "iter/sec",
            "range": "stddev: 0.000049670381648333634",
            "extra": "mean: 2.2901188840600626 msec\nrounds: 414"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 300.88628588147645,
            "unit": "iter/sec",
            "range": "stddev: 0.000117974619438723",
            "extra": "mean: 3.323514719424317 msec\nrounds: 278"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 323.7243687274925,
            "unit": "iter/sec",
            "range": "stddev: 0.00010325465549473638",
            "extra": "mean: 3.0890476485623752 msec\nrounds: 313"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 413.17732748801336,
            "unit": "iter/sec",
            "range": "stddev: 0.00007440757324733767",
            "extra": "mean: 2.420268329048164 msec\nrounds: 389"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 258.4988563688107,
            "unit": "iter/sec",
            "range": "stddev: 0.00006482626404709187",
            "extra": "mean: 3.8684890681808652 msec\nrounds: 44"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 268.08734371614054,
            "unit": "iter/sec",
            "range": "stddev: 0.00017488579194183324",
            "extra": "mean: 3.7301275999766403 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 418.2972215798135,
            "unit": "iter/sec",
            "range": "stddev: 0.00004423335721738846",
            "extra": "mean: 2.390644614428055 msec\nrounds: 402"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 256.08751145367296,
            "unit": "iter/sec",
            "range": "stddev: 0.0002387421990794776",
            "extra": "mean: 3.9049151374994056 msec\nrounds: 240"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 265.26049875900117,
            "unit": "iter/sec",
            "range": "stddev: 0.00003309185695004946",
            "extra": "mean: 3.769879061067952 msec\nrounds: 262"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 400.44128035880345,
            "unit": "iter/sec",
            "range": "stddev: 0.000036303536132381676",
            "extra": "mean: 2.4972450370350927 msec\nrounds: 378"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 225.45233798923505,
            "unit": "iter/sec",
            "range": "stddev: 0.000049435497962332966",
            "extra": "mean: 4.435527299999649 msec\nrounds: 50"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 229.3288269916855,
            "unit": "iter/sec",
            "range": "stddev: 0.0001516848658267915",
            "extra": "mean: 4.3605507999927795 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 403.4394041099784,
            "unit": "iter/sec",
            "range": "stddev: 0.00003375028867057649",
            "extra": "mean: 2.4786869844954413 msec\nrounds: 387"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 224.90602679184622,
            "unit": "iter/sec",
            "range": "stddev: 0.00009361822695396368",
            "extra": "mean: 4.446301480953707 msec\nrounds: 210"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 232.86972652811696,
            "unit": "iter/sec",
            "range": "stddev: 0.0000418683212664164",
            "extra": "mean: 4.294246465219509 msec\nrounds: 230"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_to_matrix",
            "value": 608.5416851495381,
            "unit": "iter/sec",
            "range": "stddev: 0.00003520110945447548",
            "extra": "mean: 1.6432728018529545 msec\nrounds: 540"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_from_matrix",
            "value": 11063.589071195767,
            "unit": "iter/sec",
            "range": "stddev: 0.00000516453681532192",
            "extra": "mean: 90.38658192787692 usec\nrounds: 3973"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body",
            "value": 44855.11796982301,
            "unit": "iter/sec",
            "range": "stddev: 0.000009012459436589379",
            "extra": "mean: 22.293999999571195 usec\nrounds: 9"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body_vmap",
            "value": 34684.3722175915,
            "unit": "iter/sec",
            "range": "stddev: 0.000010442979020362166",
            "extra": "mean: 28.83142856749795 usec\nrounds: 7"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body",
            "value": 369.42636287047793,
            "unit": "iter/sec",
            "range": "stddev: 0.00009041012157076831",
            "extra": "mean: 2.7068994000046587 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body_matrix",
            "value": 4.044771023607339,
            "unit": "iter/sec",
            "range": "stddev: 0.0004982390516789901",
            "extra": "mean: 247.23278379999556 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=1,]",
            "value": 561.6400253814886,
            "unit": "iter/sec",
            "range": "stddev: 0.000023187300849103876",
            "extra": "mean: 1.7804998839261137 msec\nrounds: 112"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.638039636595854,
            "unit": "iter/sec",
            "range": "stddev: 0.0006631375365949224",
            "extra": "mean: 215.6083341999988 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=1,]",
            "value": 1167.8302093068244,
            "unit": "iter/sec",
            "range": "stddev: 0.00001687796149330784",
            "extra": "mean: 856.288861198032 usec\nrounds: 951"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=10000,]",
            "value": 16.66911230325971,
            "unit": "iter/sec",
            "range": "stddev: 0.001301419361307711",
            "extra": "mean: 59.991197000001385 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 572.5337250996744,
            "unit": "iter/sec",
            "range": "stddev: 0.000021864072399036184",
            "extra": "mean: 1.7466219999981776 msec\nrounds: 496"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.61672481082201,
            "unit": "iter/sec",
            "range": "stddev: 0.0005847425234568803",
            "extra": "mean: 216.60377020001533 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1160.7322455091844,
            "unit": "iter/sec",
            "range": "stddev: 0.00001823879868592156",
            "extra": "mean: 861.5251311135281 usec\nrounds: 961"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 16.511348715203535,
            "unit": "iter/sec",
            "range": "stddev: 0.0010910411017928572",
            "extra": "mean: 60.56440435294101 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=1,]",
            "value": 563.0599191009965,
            "unit": "iter/sec",
            "range": "stddev: 0.000026272025845103325",
            "extra": "mean: 1.776009916665067 msec\nrounds: 516"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.648718555022749,
            "unit": "iter/sec",
            "range": "stddev: 0.0005770474141473796",
            "extra": "mean: 215.1130442000067 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=1,]",
            "value": 1163.9100025771488,
            "unit": "iter/sec",
            "range": "stddev: 0.00001774548501289175",
            "extra": "mean: 859.1729582062044 usec\nrounds: 981"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=10000,]",
            "value": 16.935799987690785,
            "unit": "iter/sec",
            "range": "stddev: 0.0012211294841023884",
            "extra": "mean: 59.046516888887226 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 564.5085965868695,
            "unit": "iter/sec",
            "range": "stddev: 0.000024828896564306785",
            "extra": "mean: 1.7714522082501443 msec\nrounds: 509"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.6306562632739725,
            "unit": "iter/sec",
            "range": "stddev: 0.0013932505576852918",
            "extra": "mean: 215.95211199999085 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1157.7221119589208,
            "unit": "iter/sec",
            "range": "stddev: 0.000027992388232708716",
            "extra": "mean: 863.7651381711563 usec\nrounds: 1006"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 16.89865158760511,
            "unit": "iter/sec",
            "range": "stddev: 0.0002804262340950683",
            "extra": "mean: 59.17631917646518 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/orbits/tests/test_benchmarks.py::test_benchmark_iterate_real_orbits",
            "value": 17088.206842846605,
            "unit": "iter/sec",
            "range": "stddev: 0.000004667865268343822",
            "extra": "mean: 58.51989089297663 usec\nrounds: 14692"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1389.0743421002321,
            "unit": "iter/sec",
            "range": "stddev: 0.000017259844033259478",
            "extra": "mean: 719.9038738906046 usec\nrounds: 1015"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 16.77428743601618,
            "unit": "iter/sec",
            "range": "stddev: 0.00020256454001812583",
            "extra": "mean: 59.615050941174026 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1415.4848284450045,
            "unit": "iter/sec",
            "range": "stddev: 0.000030349194237252598",
            "extra": "mean: 706.4717190212207 usec\nrounds: 1267"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 18.429497507215892,
            "unit": "iter/sec",
            "range": "stddev: 0.0002838925166716696",
            "extra": "mean: 54.26083915790215 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1407.7494370785837,
            "unit": "iter/sec",
            "range": "stddev: 0.000014915559548481724",
            "extra": "mean: 710.3536848682667 usec\nrounds: 1282"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 17.62212434685534,
            "unit": "iter/sec",
            "range": "stddev: 0.00016687681528858357",
            "extra": "mean: 56.7468473333324 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1410.3327765350527,
            "unit": "iter/sec",
            "range": "stddev: 0.000014924085275229059",
            "extra": "mean: 709.0525134478046 usec\nrounds: 1227"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 16.656438764114547,
            "unit": "iter/sec",
            "range": "stddev: 0.0001791648567786974",
            "extra": "mean: 60.03684305882055 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1421.0580929413413,
            "unit": "iter/sec",
            "range": "stddev: 0.00003383191678814205",
            "extra": "mean: 703.7009992534332 usec\nrounds: 1339"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 18.358474278295247,
            "unit": "iter/sec",
            "range": "stddev: 0.00018658659867855582",
            "extra": "mean: 54.470757473690185 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1378.1432566642375,
            "unit": "iter/sec",
            "range": "stddev: 0.00007415355929159617",
            "extra": "mean: 725.6139702199581 usec\nrounds: 1276"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 17.276921048109028,
            "unit": "iter/sec",
            "range": "stddev: 0.0010948472114892427",
            "extra": "mean: 57.88068355556043 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1404.0671209736845,
            "unit": "iter/sec",
            "range": "stddev: 0.000015448521003676245",
            "extra": "mean: 712.2166633362411 usec\nrounds: 1301"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 17.04705604572248,
            "unit": "iter/sec",
            "range": "stddev: 0.0002697899234002786",
            "extra": "mean: 58.66115517646369 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1409.7156048469417,
            "unit": "iter/sec",
            "range": "stddev: 0.00003238896202103806",
            "extra": "mean: 709.3629357309795 usec\nrounds: 1307"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 17.480624868709416,
            "unit": "iter/sec",
            "range": "stddev: 0.00021764309253761215",
            "extra": "mean: 57.206193000000546 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1418.0398265403226,
            "unit": "iter/sec",
            "range": "stddev: 0.000032317412029376325",
            "extra": "mean: 705.1988112631225 usec\nrounds: 1314"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 18.303593954118316,
            "unit": "iter/sec",
            "range": "stddev: 0.0010997571569809179",
            "extra": "mean: 54.63407910526772 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1411.3967276408928,
            "unit": "iter/sec",
            "range": "stddev: 0.00001892448021564643",
            "extra": "mean: 708.5180094412362 usec\nrounds: 1271"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 16.89537725175862,
            "unit": "iter/sec",
            "range": "stddev: 0.0008151986930201781",
            "extra": "mean: 59.18778758822394 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1420.0655749348882,
            "unit": "iter/sec",
            "range": "stddev: 0.000014579053278497532",
            "extra": "mean: 704.1928328174924 usec\nrounds: 1292"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 17.3430178170796,
            "unit": "iter/sec",
            "range": "stddev: 0.00023202284602135044",
            "extra": "mean: 57.66009183333646 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1420.9738199907747,
            "unit": "iter/sec",
            "range": "stddev: 0.000028858041986033032",
            "extra": "mean: 703.74273328026 usec\nrounds: 1286"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 18.351482891961773,
            "unit": "iter/sec",
            "range": "stddev: 0.0004012898647808757",
            "extra": "mean: 54.49150926315688 msec\nrounds: 19"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "akoumjian@users.noreply.github.com",
            "name": "Alec Koumjian",
            "username": "akoumjian"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "3d2a90dabb88ef0a7a9a73d8e76e1fb7f1f56c1a",
          "message": "Ak/more perf improvements (#130)\n\n* this seems like a better chunk size\r\n\r\n* Change chunking",
          "timestamp": "2024-12-09T10:07:50-05:00",
          "tree_id": "4e1a0b418c53f0b92581c519c6d0805ee4c851c1",
          "url": "https://github.com/B612-Asteroid-Institute/adam_core/commit/3d2a90dabb88ef0a7a9a73d8e76e1fb7f1f56c1a"
        },
        "date": 1733757148298,
        "tool": "pytest",
        "benches": [
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 432.30451703115165,
            "unit": "iter/sec",
            "range": "stddev: 0.00011164682831696494",
            "extra": "mean: 2.313184249999267 msec\nrounds: 8"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 294.5708356248617,
            "unit": "iter/sec",
            "range": "stddev: 0.00019641552542132203",
            "extra": "mean: 3.3947692000083407 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 322.2904797656551,
            "unit": "iter/sec",
            "range": "stddev: 0.00015456530584201782",
            "extra": "mean: 3.1027909999920666 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 436.9647577244637,
            "unit": "iter/sec",
            "range": "stddev: 0.00004331397408696329",
            "extra": "mean: 2.288514078818614 msec\nrounds: 406"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 307.86089976370465,
            "unit": "iter/sec",
            "range": "stddev: 0.000025930693613758257",
            "extra": "mean: 3.2482202214296763 msec\nrounds: 280"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 331.73124956934424,
            "unit": "iter/sec",
            "range": "stddev: 0.00005997348523250627",
            "extra": "mean: 3.0144883887128713 msec\nrounds: 319"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 420.40046112743784,
            "unit": "iter/sec",
            "range": "stddev: 0.000039063819774274005",
            "extra": "mean: 2.3786843556693094 msec\nrounds: 388"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 261.6815762762976,
            "unit": "iter/sec",
            "range": "stddev: 0.000059592349988673223",
            "extra": "mean: 3.821438307694026 msec\nrounds: 52"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 264.055446996967,
            "unit": "iter/sec",
            "range": "stddev: 0.0001524297482406814",
            "extra": "mean: 3.7870833999932074 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 410.6810938631765,
            "unit": "iter/sec",
            "range": "stddev: 0.00007277184332856659",
            "extra": "mean: 2.4349793914135294 msec\nrounds: 396"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 257.51360098189735,
            "unit": "iter/sec",
            "range": "stddev: 0.000024477021675666605",
            "extra": "mean: 3.8832900327866477 msec\nrounds: 244"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 267.75268394677875,
            "unit": "iter/sec",
            "range": "stddev: 0.00002717670296936621",
            "extra": "mean: 3.7347898264159705 msec\nrounds: 265"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 388.81277195579185,
            "unit": "iter/sec",
            "range": "stddev: 0.000028752595255949453",
            "extra": "mean: 2.571931973761655 msec\nrounds: 343"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 228.0691122925848,
            "unit": "iter/sec",
            "range": "stddev: 0.00006360151830342465",
            "extra": "mean: 4.384635823535465 msec\nrounds: 51"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 233.68805934375067,
            "unit": "iter/sec",
            "range": "stddev: 0.0001327196534628909",
            "extra": "mean: 4.279208800005563 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 404.5398890993521,
            "unit": "iter/sec",
            "range": "stddev: 0.000038176855820553214",
            "extra": "mean: 2.471944119593129 msec\nrounds: 393"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 228.39633003541638,
            "unit": "iter/sec",
            "range": "stddev: 0.00007483191333768143",
            "extra": "mean: 4.378354064817656 msec\nrounds: 216"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 235.52271562297528,
            "unit": "iter/sec",
            "range": "stddev: 0.000029795413843229774",
            "extra": "mean: 4.245874956710332 msec\nrounds: 231"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_to_matrix",
            "value": 649.5332182237352,
            "unit": "iter/sec",
            "range": "stddev: 0.000017260789468664565",
            "extra": "mean: 1.539567141361421 msec\nrounds: 573"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_from_matrix",
            "value": 10908.748970262608,
            "unit": "iter/sec",
            "range": "stddev: 0.0000054368848480825566",
            "extra": "mean: 91.66953999271713 usec\nrounds: 2913"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body",
            "value": 45449.26598226524,
            "unit": "iter/sec",
            "range": "stddev: 0.000009177802314834792",
            "extra": "mean: 22.002555561407966 usec\nrounds: 9"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body_vmap",
            "value": 33293.06458427732,
            "unit": "iter/sec",
            "range": "stddev: 0.0000117079549625616",
            "extra": "mean: 30.036285709554377 usec\nrounds: 7"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body",
            "value": 378.07059485315256,
            "unit": "iter/sec",
            "range": "stddev: 0.0000954384411347726",
            "extra": "mean: 2.6450086666708708 msec\nrounds: 6"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body_matrix",
            "value": 4.026333992678144,
            "unit": "iter/sec",
            "range": "stddev: 0.0017457416330079487",
            "extra": "mean: 248.36489019999135 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=1,]",
            "value": 570.1048276400348,
            "unit": "iter/sec",
            "range": "stddev: 0.00003881811701147336",
            "extra": "mean: 1.7540633783781985 msec\nrounds: 111"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.75988823751963,
            "unit": "iter/sec",
            "range": "stddev: 0.0025360954836419524",
            "extra": "mean: 210.08896639999648 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=1,]",
            "value": 1179.1747881604874,
            "unit": "iter/sec",
            "range": "stddev: 0.000016893756183096502",
            "extra": "mean: 848.0506961652394 usec\nrounds: 1017"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=10000,]",
            "value": 17.530282752681856,
            "unit": "iter/sec",
            "range": "stddev: 0.00045906024298829194",
            "extra": "mean: 57.044145499992915 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 570.1633948937412,
            "unit": "iter/sec",
            "range": "stddev: 0.00002377643134675274",
            "extra": "mean: 1.7538832007732899 msec\nrounds: 518"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.759276286133919,
            "unit": "iter/sec",
            "range": "stddev: 0.0004970194296542068",
            "extra": "mean: 210.11597980001397 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1175.296745805707,
            "unit": "iter/sec",
            "range": "stddev: 0.000017259940366192394",
            "extra": "mean: 850.8489482070888 usec\nrounds: 1004"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 17.461801080534652,
            "unit": "iter/sec",
            "range": "stddev: 0.00022983063569287284",
            "extra": "mean: 57.26786116666619 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=1,]",
            "value": 574.5966943422288,
            "unit": "iter/sec",
            "range": "stddev: 0.000023506211261726606",
            "extra": "mean: 1.7403511190483838 msec\nrounds: 504"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.770432009073559,
            "unit": "iter/sec",
            "range": "stddev: 0.00034210513344102835",
            "extra": "mean: 209.6246205999705 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=1,]",
            "value": 1179.3274618475116,
            "unit": "iter/sec",
            "range": "stddev: 0.00001651493664137909",
            "extra": "mean: 847.9409089935202 usec\nrounds: 934"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=10000,]",
            "value": 17.82066981553893,
            "unit": "iter/sec",
            "range": "stddev: 0.000281170049462807",
            "extra": "mean: 56.11461355554878 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 570.8790585649443,
            "unit": "iter/sec",
            "range": "stddev: 0.00004046564053636006",
            "extra": "mean: 1.7516845030430173 msec\nrounds: 493"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.760368205721113,
            "unit": "iter/sec",
            "range": "stddev: 0.0008159340761819243",
            "extra": "mean: 210.06778400002304 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1180.3710423218765,
            "unit": "iter/sec",
            "range": "stddev: 0.000017062495127488214",
            "extra": "mean: 847.1912340656262 usec\nrounds: 957"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 17.511137034511258,
            "unit": "iter/sec",
            "range": "stddev: 0.0013968067501714061",
            "extra": "mean: 57.10651444444655 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/orbits/tests/test_benchmarks.py::test_benchmark_iterate_real_orbits",
            "value": 19281.093760539086,
            "unit": "iter/sec",
            "range": "stddev: 0.000005529217477658873",
            "extra": "mean: 51.86427764002744 usec\nrounds: 15340"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1422.6807766617194,
            "unit": "iter/sec",
            "range": "stddev: 0.000058073912820435956",
            "extra": "mean: 702.8983707409556 usec\nrounds: 998"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 17.548287273560536,
            "unit": "iter/sec",
            "range": "stddev: 0.00018595715132761715",
            "extra": "mean: 56.98561827778311 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1456.0879328734345,
            "unit": "iter/sec",
            "range": "stddev: 0.000014527202376214345",
            "extra": "mean: 686.7717102954122 usec\nrounds: 1253"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 19.31504944612711,
            "unit": "iter/sec",
            "range": "stddev: 0.0004798875356327217",
            "extra": "mean: 51.77310070000942 msec\nrounds: 20"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1463.222035565179,
            "unit": "iter/sec",
            "range": "stddev: 0.00002998083342222413",
            "extra": "mean: 683.4232780083464 usec\nrounds: 1205"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 18.307567625597564,
            "unit": "iter/sec",
            "range": "stddev: 0.000205213941571938",
            "extra": "mean: 54.62222073683913 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1407.5313343693438,
            "unit": "iter/sec",
            "range": "stddev: 0.00003407999540297178",
            "extra": "mean: 710.4637570630414 usec\nrounds: 1239"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 16.1954263225321,
            "unit": "iter/sec",
            "range": "stddev: 0.010604872113258921",
            "extra": "mean: 61.74582749999836 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1432.7128153076978,
            "unit": "iter/sec",
            "range": "stddev: 0.00001579512723792388",
            "extra": "mean: 697.9765863162423 usec\nrounds: 1257"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 19.234563100384683,
            "unit": "iter/sec",
            "range": "stddev: 0.00025973540098919143",
            "extra": "mean: 51.98974339999438 msec\nrounds: 20"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1402.3781938813192,
            "unit": "iter/sec",
            "range": "stddev: 0.00008166559020797571",
            "extra": "mean: 713.0744077190267 usec\nrounds: 1192"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 18.106368907926786,
            "unit": "iter/sec",
            "range": "stddev: 0.0002745046808976119",
            "extra": "mean: 55.2291851052593 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1434.6685276114665,
            "unit": "iter/sec",
            "range": "stddev: 0.000016096492877189955",
            "extra": "mean: 697.025118174766 usec\nrounds: 1227"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 17.809732941679076,
            "unit": "iter/sec",
            "range": "stddev: 0.0002653179448812569",
            "extra": "mean: 56.149073277778264 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1436.3856922351115,
            "unit": "iter/sec",
            "range": "stddev: 0.000017013277614023493",
            "extra": "mean: 696.1918413737008 usec\nrounds: 1223"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 18.204277539383163,
            "unit": "iter/sec",
            "range": "stddev: 0.00021296903704593351",
            "extra": "mean: 54.93214426316005 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1442.9296161686068,
            "unit": "iter/sec",
            "range": "stddev: 0.00003126210962583405",
            "extra": "mean: 693.0344964817394 usec\nrounds: 1279"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 19.206897070740542,
            "unit": "iter/sec",
            "range": "stddev: 0.0001424328220793406",
            "extra": "mean: 52.06463055000086 msec\nrounds: 20"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1426.4729952314626,
            "unit": "iter/sec",
            "range": "stddev: 0.00003348148581180589",
            "extra": "mean: 701.0297449323517 usec\nrounds: 1184"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 17.738889384356472,
            "unit": "iter/sec",
            "range": "stddev: 0.00029223160678386166",
            "extra": "mean: 56.37331505555684 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1426.9535378004218,
            "unit": "iter/sec",
            "range": "stddev: 0.00003108647147500599",
            "extra": "mean: 700.7936653224538 usec\nrounds: 1240"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 17.808748290361358,
            "unit": "iter/sec",
            "range": "stddev: 0.0015890318203003895",
            "extra": "mean: 56.15217777777401 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1441.5577738629786,
            "unit": "iter/sec",
            "range": "stddev: 0.00001970370189295064",
            "extra": "mean: 693.6940149962043 usec\nrounds: 1267"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 18.972216844702828,
            "unit": "iter/sec",
            "range": "stddev: 0.0020014816420898885",
            "extra": "mean: 52.70865330000731 msec\nrounds: 20"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "akoumjian@users.noreply.github.com",
            "name": "Alec Koumjian",
            "username": "akoumjian"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "16d9421f4bba62941e185a9790731f6c47534cfa",
          "message": "Ensure some time scales and coordinate origins are correct (#131)",
          "timestamp": "2024-12-10T13:33:56-05:00",
          "tree_id": "cf749772d4450b083ee8cde7186c2c9cb27e3151",
          "url": "https://github.com/B612-Asteroid-Institute/adam_core/commit/16d9421f4bba62941e185a9790731f6c47534cfa"
        },
        "date": 1733855884359,
        "tool": "pytest",
        "benches": [
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 419.6170553748794,
            "unit": "iter/sec",
            "range": "stddev: 0.00009839735400155007",
            "extra": "mean: 2.3831252500130518 msec\nrounds: 8"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 278.60829362186064,
            "unit": "iter/sec",
            "range": "stddev: 0.00026607984754047385",
            "extra": "mean: 3.589268600012474 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 286.0280994006964,
            "unit": "iter/sec",
            "range": "stddev: 0.00018859868027226818",
            "extra": "mean: 3.496159999997417 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 420.8803084127765,
            "unit": "iter/sec",
            "range": "stddev: 0.00005748341170054588",
            "extra": "mean: 2.3759724083343294 msec\nrounds: 360"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 280.49531683166373,
            "unit": "iter/sec",
            "range": "stddev: 0.0002852187823935135",
            "extra": "mean: 3.565121911108909 msec\nrounds: 270"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 314.70775100169624,
            "unit": "iter/sec",
            "range": "stddev: 0.00009162964033167583",
            "extra": "mean: 3.177551225913753 msec\nrounds: 301"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 408.5935984667832,
            "unit": "iter/sec",
            "range": "stddev: 0.0000697756700993536",
            "extra": "mean: 2.447419645712573 msec\nrounds: 350"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 245.26414089567643,
            "unit": "iter/sec",
            "range": "stddev: 0.00015214554081657524",
            "extra": "mean: 4.07723687754808 msec\nrounds: 49"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 248.81017729153731,
            "unit": "iter/sec",
            "range": "stddev: 0.00019593670375158795",
            "extra": "mean: 4.019128200002342 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 408.0725459284117,
            "unit": "iter/sec",
            "range": "stddev: 0.00006427173108269423",
            "extra": "mean: 2.450544664123105 msec\nrounds: 393"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 242.8373396717236,
            "unit": "iter/sec",
            "range": "stddev: 0.0001381167027123985",
            "extra": "mean: 4.117982849556154 msec\nrounds: 226"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 249.30821226913562,
            "unit": "iter/sec",
            "range": "stddev: 0.00013980911759729837",
            "extra": "mean: 4.011099317179614 msec\nrounds: 227"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 393.86966886810706,
            "unit": "iter/sec",
            "range": "stddev: 0.00007426258411856389",
            "extra": "mean: 2.538910911504751 msec\nrounds: 339"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 218.70598120415454,
            "unit": "iter/sec",
            "range": "stddev: 0.00010422927359077603",
            "extra": "mean: 4.57234865957568 msec\nrounds: 47"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 220.88236940355375,
            "unit": "iter/sec",
            "range": "stddev: 0.00019975622024792007",
            "extra": "mean: 4.527296599997044 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 401.74184984948846,
            "unit": "iter/sec",
            "range": "stddev: 0.00006588763770560638",
            "extra": "mean: 2.4891606397855925 msec\nrounds: 372"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 212.1935301493201,
            "unit": "iter/sec",
            "range": "stddev: 0.00019299479650319257",
            "extra": "mean: 4.712679030771119 msec\nrounds: 195"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 223.81191125497313,
            "unit": "iter/sec",
            "range": "stddev: 0.00012124950493599001",
            "extra": "mean: 4.468037444444905 msec\nrounds: 207"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_to_matrix",
            "value": 595.8222374680348,
            "unit": "iter/sec",
            "range": "stddev: 0.000054594339723611174",
            "extra": "mean: 1.6783529333338265 msec\nrounds: 495"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_from_matrix",
            "value": 10783.98014805846,
            "unit": "iter/sec",
            "range": "stddev: 0.000008141067801561719",
            "extra": "mean: 92.73014103053957 usec\nrounds: 2659"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body",
            "value": 40428.74688696914,
            "unit": "iter/sec",
            "range": "stddev: 0.00001333835494149915",
            "extra": "mean: 24.734874983778354 usec\nrounds: 8"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body_vmap",
            "value": 28967.59490926491,
            "unit": "iter/sec",
            "range": "stddev: 0.000015920303583393626",
            "extra": "mean: 34.521333342733364 usec\nrounds: 6"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body",
            "value": 337.77457357320515,
            "unit": "iter/sec",
            "range": "stddev: 0.00009490275060898878",
            "extra": "mean: 2.9605543999991824 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body_matrix",
            "value": 3.896806937768496,
            "unit": "iter/sec",
            "range": "stddev: 0.000776770092504826",
            "extra": "mean: 256.6203601999973 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=1,]",
            "value": 550.9213744249554,
            "unit": "iter/sec",
            "range": "stddev: 0.000057789644860724225",
            "extra": "mean: 1.8151410462950126 msec\nrounds: 108"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.501179673669926,
            "unit": "iter/sec",
            "range": "stddev: 0.002752049697981254",
            "extra": "mean: 222.1639820000064 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=1,]",
            "value": 1133.879935815382,
            "unit": "iter/sec",
            "range": "stddev: 0.00002921858111454551",
            "extra": "mean: 881.9275907557991 usec\nrounds: 887"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=10000,]",
            "value": 16.244365387451868,
            "unit": "iter/sec",
            "range": "stddev: 0.00033052975161478797",
            "extra": "mean: 61.55980711763973 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 557.8496321058556,
            "unit": "iter/sec",
            "range": "stddev: 0.00006340434014511374",
            "extra": "mean: 1.7925977583332773 msec\nrounds: 480"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.531114815180872,
            "unit": "iter/sec",
            "range": "stddev: 0.0010618084001033415",
            "extra": "mean: 220.6962393999902 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1155.579225427174,
            "unit": "iter/sec",
            "range": "stddev: 0.00003160887158853436",
            "extra": "mean: 865.3668896049406 usec\nrounds: 933"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 16.345260011896944,
            "unit": "iter/sec",
            "range": "stddev: 0.001050898656476228",
            "extra": "mean: 61.17981599999922 msec\nrounds: 16"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=1,]",
            "value": 563.1446254768066,
            "unit": "iter/sec",
            "range": "stddev: 0.000028417956935644887",
            "extra": "mean: 1.7757427750523485 msec\nrounds: 489"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.585994423477036,
            "unit": "iter/sec",
            "range": "stddev: 0.0006611733945207942",
            "extra": "mean: 218.05521499998122 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=1,]",
            "value": 1157.631105479008,
            "unit": "iter/sec",
            "range": "stddev: 0.000018693184333983453",
            "extra": "mean: 863.8330425530652 usec\nrounds: 940"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=10000,]",
            "value": 16.73640221737796,
            "unit": "iter/sec",
            "range": "stddev: 0.00039124629827658506",
            "extra": "mean: 59.74999805882215 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 561.690064249194,
            "unit": "iter/sec",
            "range": "stddev: 0.000029574740323530068",
            "extra": "mean: 1.7803412658486149 msec\nrounds: 489"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.52229677225443,
            "unit": "iter/sec",
            "range": "stddev: 0.0019125223732980785",
            "extra": "mean: 221.12657579999677 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1138.4911862138051,
            "unit": "iter/sec",
            "range": "stddev: 0.000026307925760368623",
            "extra": "mean: 878.3555042930329 usec\nrounds: 932"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 16.4388927849494,
            "unit": "iter/sec",
            "range": "stddev: 0.00044205577521007133",
            "extra": "mean: 60.83134752941198 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/orbits/tests/test_benchmarks.py::test_benchmark_iterate_real_orbits",
            "value": 17080.86634582156,
            "unit": "iter/sec",
            "range": "stddev: 0.000002522801029265488",
            "extra": "mean: 58.54503979797412 usec\nrounds: 14272"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1362.677472213948,
            "unit": "iter/sec",
            "range": "stddev: 0.0000389608397073679",
            "extra": "mean: 733.8493666995871 usec\nrounds: 1009"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 16.257263335241408,
            "unit": "iter/sec",
            "range": "stddev: 0.00020559115019123725",
            "extra": "mean: 61.51096770587869 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1396.2056188069776,
            "unit": "iter/sec",
            "range": "stddev: 0.000035617746764091666",
            "extra": "mean: 716.2268841565577 usec\nrounds: 1174"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 17.94969173765949,
            "unit": "iter/sec",
            "range": "stddev: 0.00032682562020323267",
            "extra": "mean: 55.711263157903836 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1393.751307757337,
            "unit": "iter/sec",
            "range": "stddev: 0.00002095468691352062",
            "extra": "mean: 717.4881160176876 usec\nrounds: 1155"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 16.964429026685963,
            "unit": "iter/sec",
            "range": "stddev: 0.0003086292107581421",
            "extra": "mean: 58.94687044444266 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1388.628225519953,
            "unit": "iter/sec",
            "range": "stddev: 0.000021665197917707776",
            "extra": "mean: 720.1351532556985 usec\nrounds: 1044"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 16.23138104843908,
            "unit": "iter/sec",
            "range": "stddev: 0.0002531403924689621",
            "extra": "mean: 61.60905205883062 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1387.6227247704132,
            "unit": "iter/sec",
            "range": "stddev: 0.000027298706115731185",
            "extra": "mean: 720.6569784056061 usec\nrounds: 1204"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 18.00247584639329,
            "unit": "iter/sec",
            "range": "stddev: 0.00030916114375747003",
            "extra": "mean: 55.547915105262895 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1387.1842342511093,
            "unit": "iter/sec",
            "range": "stddev: 0.000022055471773223082",
            "extra": "mean: 720.8847788987913 usec\nrounds: 1090"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 16.95819462469584,
            "unit": "iter/sec",
            "range": "stddev: 0.00023557130230704256",
            "extra": "mean: 58.96854129411408 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1390.7419192356444,
            "unit": "iter/sec",
            "range": "stddev: 0.000018908379918698653",
            "extra": "mean: 719.0406689902629 usec\nrounds: 1148"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 16.57307576190747,
            "unit": "iter/sec",
            "range": "stddev: 0.0006515856257849063",
            "extra": "mean: 60.33882994117836 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1397.5405546688332,
            "unit": "iter/sec",
            "range": "stddev: 0.000020956801587749845",
            "extra": "mean: 715.5427416107891 usec\nrounds: 1192"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 17.181867560653043,
            "unit": "iter/sec",
            "range": "stddev: 0.00038043771193250716",
            "extra": "mean: 58.200890937492034 msec\nrounds: 16"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1424.1213714917765,
            "unit": "iter/sec",
            "range": "stddev: 0.000015763944728106378",
            "extra": "mean: 702.1873416255902 usec\nrounds: 1206"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 18.273441005179645,
            "unit": "iter/sec",
            "range": "stddev: 0.00026393739416541906",
            "extra": "mean: 54.7242306315788 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1415.2577869693735,
            "unit": "iter/sec",
            "range": "stddev: 0.00001557669709253375",
            "extra": "mean: 706.5850541203489 usec\nrounds: 1201"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 16.693866556274813,
            "unit": "iter/sec",
            "range": "stddev: 0.0002672910466217177",
            "extra": "mean: 59.902239941179154 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1403.6919289967632,
            "unit": "iter/sec",
            "range": "stddev: 0.000024372077292372314",
            "extra": "mean: 712.4070313025971 usec\nrounds: 1182"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 17.158775567735116,
            "unit": "iter/sec",
            "range": "stddev: 0.0002719459403230136",
            "extra": "mean: 58.27921672222184 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1425.701463268087,
            "unit": "iter/sec",
            "range": "stddev: 0.000014369173246809701",
            "extra": "mean: 701.4091138741865 usec\nrounds: 1247"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 18.258390657357143,
            "unit": "iter/sec",
            "range": "stddev: 0.00032596605259374036",
            "extra": "mean: 54.769339684220974 msec\nrounds: 19"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "akoumjian@users.noreply.github.com",
            "name": "Alec Koumjian",
            "username": "akoumjian"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "16d9421f4bba62941e185a9790731f6c47534cfa",
          "message": "Ensure some time scales and coordinate origins are correct (#131)",
          "timestamp": "2024-12-10T13:33:56-05:00",
          "tree_id": "cf749772d4450b083ee8cde7186c2c9cb27e3151",
          "url": "https://github.com/B612-Asteroid-Institute/adam_core/commit/16d9421f4bba62941e185a9790731f6c47534cfa"
        },
        "date": 1733855886447,
        "tool": "pytest",
        "benches": [
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 440.8813880260027,
            "unit": "iter/sec",
            "range": "stddev: 0.00008842208270532047",
            "extra": "mean: 2.268183750004482 msec\nrounds: 8"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 293.3062756330693,
            "unit": "iter/sec",
            "range": "stddev: 0.00020124096044040973",
            "extra": "mean: 3.409405400009291 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 317.6671944392813,
            "unit": "iter/sec",
            "range": "stddev: 0.0001567034389383326",
            "extra": "mean: 3.1479485999966528 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 441.65824933379,
            "unit": "iter/sec",
            "range": "stddev: 0.00005973756274861252",
            "extra": "mean: 2.2641940946612653 msec\nrounds: 412"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 298.8671983449959,
            "unit": "iter/sec",
            "range": "stddev: 0.00015704487894391056",
            "extra": "mean: 3.34596772592506 msec\nrounds: 270"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 324.6181158690973,
            "unit": "iter/sec",
            "range": "stddev: 0.00008166972315692588",
            "extra": "mean: 3.0805428012626734 msec\nrounds: 317"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 409.6592376460921,
            "unit": "iter/sec",
            "range": "stddev: 0.00005090759008154338",
            "extra": "mean: 2.4410532171714583 msec\nrounds: 396"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 257.28077713477165,
            "unit": "iter/sec",
            "range": "stddev: 0.00008552693051070864",
            "extra": "mean: 3.8868041799958064 msec\nrounds: 50"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 265.07299129522085,
            "unit": "iter/sec",
            "range": "stddev: 0.00015043350864619304",
            "extra": "mean: 3.772545799984073 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 412.4085395197432,
            "unit": "iter/sec",
            "range": "stddev: 0.0000826641804076833",
            "extra": "mean: 2.4247800522378054 msec\nrounds: 402"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 258.38015340758585,
            "unit": "iter/sec",
            "range": "stddev: 0.00005655955020774578",
            "extra": "mean: 3.870266298752963 msec\nrounds: 241"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 267.606346851155,
            "unit": "iter/sec",
            "range": "stddev: 0.00007634137643806421",
            "extra": "mean: 3.7368321482905964 msec\nrounds: 263"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 396.8782899822016,
            "unit": "iter/sec",
            "range": "stddev: 0.00008958178671185646",
            "extra": "mean: 2.5196641520624525 msec\nrounds: 388"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 226.9577485531786,
            "unit": "iter/sec",
            "range": "stddev: 0.00006664527462902324",
            "extra": "mean: 4.406106450979749 msec\nrounds: 51"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 234.38610037958142,
            "unit": "iter/sec",
            "range": "stddev: 0.000166515380112634",
            "extra": "mean: 4.266464599993469 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 406.1524847662961,
            "unit": "iter/sec",
            "range": "stddev: 0.00004855309795423447",
            "extra": "mean: 2.462129464935834 msec\nrounds: 385"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 219.3674309334062,
            "unit": "iter/sec",
            "range": "stddev: 0.00022736185256630044",
            "extra": "mean: 4.558561841860526 msec\nrounds: 215"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 228.6871901459629,
            "unit": "iter/sec",
            "range": "stddev: 0.0000712682499625294",
            "extra": "mean: 4.372785372725668 msec\nrounds: 220"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_to_matrix",
            "value": 644.6238567817701,
            "unit": "iter/sec",
            "range": "stddev: 0.000019743647237097997",
            "extra": "mean: 1.551292260563261 msec\nrounds: 568"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_from_matrix",
            "value": 10777.795915996934,
            "unit": "iter/sec",
            "range": "stddev: 0.000005522117911464615",
            "extra": "mean: 92.7833490069849 usec\nrounds: 2871"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body",
            "value": 43878.03854583232,
            "unit": "iter/sec",
            "range": "stddev: 0.000009042112605231",
            "extra": "mean: 22.79044444877501 usec\nrounds: 9"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body_vmap",
            "value": 34291.87280939709,
            "unit": "iter/sec",
            "range": "stddev: 0.000010147079878976736",
            "extra": "mean: 29.161428585666727 usec\nrounds: 7"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body",
            "value": 337.45658410114606,
            "unit": "iter/sec",
            "range": "stddev: 0.00007049274904757485",
            "extra": "mean: 2.9633441666684726 msec\nrounds: 6"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body_matrix",
            "value": 3.9599550715537295,
            "unit": "iter/sec",
            "range": "stddev: 0.0008649169415056774",
            "extra": "mean: 252.52811759999076 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=1,]",
            "value": 564.122535525084,
            "unit": "iter/sec",
            "range": "stddev: 0.00003482331883493901",
            "extra": "mean: 1.772664513516026 msec\nrounds: 111"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.682792036492729,
            "unit": "iter/sec",
            "range": "stddev: 0.000866702311163321",
            "extra": "mean: 213.54781340000955 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=1,]",
            "value": 1164.2969153531942,
            "unit": "iter/sec",
            "range": "stddev: 0.000025720227760086945",
            "extra": "mean: 858.8874425529557 usec\nrounds: 940"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=10000,]",
            "value": 17.078764232958406,
            "unit": "iter/sec",
            "range": "stddev: 0.0003444602480970953",
            "extra": "mean: 58.5522457222175 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 561.9905247481902,
            "unit": "iter/sec",
            "range": "stddev: 0.000055771739382969643",
            "extra": "mean: 1.779389430894885 msec\nrounds: 492"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.685212790389627,
            "unit": "iter/sec",
            "range": "stddev: 0.0005436866840030911",
            "extra": "mean: 213.43747759999587 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1163.7413596497242,
            "unit": "iter/sec",
            "range": "stddev: 0.0000643371424294948",
            "extra": "mean: 859.2974647742958 usec\nrounds: 951"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 16.901283843293317,
            "unit": "iter/sec",
            "range": "stddev: 0.0012934195906635529",
            "extra": "mean: 59.167102882353824 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=1,]",
            "value": 574.6977746632316,
            "unit": "iter/sec",
            "range": "stddev: 0.00003482766454222119",
            "extra": "mean: 1.7400450185943248 msec\nrounds: 484"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.679652147305007,
            "unit": "iter/sec",
            "range": "stddev: 0.0011987770932856154",
            "extra": "mean: 213.69109679998246 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=1,]",
            "value": 1167.776525660984,
            "unit": "iter/sec",
            "range": "stddev: 0.00003745012270884351",
            "extra": "mean: 856.3282255001494 usec\nrounds: 949"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=10000,]",
            "value": 17.43040217137851,
            "unit": "iter/sec",
            "range": "stddev: 0.0004126168493188322",
            "extra": "mean: 57.37102277777871 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 567.4461466200381,
            "unit": "iter/sec",
            "range": "stddev: 0.00003528002730376115",
            "extra": "mean: 1.7622817706251865 msec\nrounds: 497"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.694778470577855,
            "unit": "iter/sec",
            "range": "stddev: 0.0005985102946143851",
            "extra": "mean: 213.00259559998267 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1160.067937413106,
            "unit": "iter/sec",
            "range": "stddev: 0.0000251623164431481",
            "extra": "mean: 862.0184799089874 usec\nrounds: 871"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 17.290200642964603,
            "unit": "iter/sec",
            "range": "stddev: 0.0006963985601179963",
            "extra": "mean: 57.836228777767296 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/orbits/tests/test_benchmarks.py::test_benchmark_iterate_real_orbits",
            "value": 19402.30017927111,
            "unit": "iter/sec",
            "range": "stddev: 0.000004547087180119796",
            "extra": "mean: 51.54028083063949 usec\nrounds: 15650"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1435.0386599277163,
            "unit": "iter/sec",
            "range": "stddev: 0.00002085773228701962",
            "extra": "mean: 696.845337985787 usec\nrounds: 1003"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 17.192550950592455,
            "unit": "iter/sec",
            "range": "stddev: 0.0011666417484373969",
            "extra": "mean: 58.16472511111215 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1464.795114774486,
            "unit": "iter/sec",
            "range": "stddev: 0.000025651439111821925",
            "extra": "mean: 682.6893330771082 usec\nrounds: 1300"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 19.10068590114432,
            "unit": "iter/sec",
            "range": "stddev: 0.0003104848325041872",
            "extra": "mean: 52.35414084999377 msec\nrounds: 20"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1440.3580525979257,
            "unit": "iter/sec",
            "range": "stddev: 0.0000184799659138364",
            "extra": "mean: 694.2718153977987 usec\nrounds: 1208"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 17.957307316548565,
            "unit": "iter/sec",
            "range": "stddev: 0.0011234844156421237",
            "extra": "mean: 55.68763636842421 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1439.0114771580659,
            "unit": "iter/sec",
            "range": "stddev: 0.000019124416160692504",
            "extra": "mean: 694.921490115507 usec\nrounds: 1214"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 16.9962503042546,
            "unit": "iter/sec",
            "range": "stddev: 0.0012130794582209683",
            "extra": "mean: 58.83650699999836 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1445.5647227216546,
            "unit": "iter/sec",
            "range": "stddev: 0.000030414399478148474",
            "extra": "mean: 691.7711703127605 usec\nrounds: 1280"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 18.9982527933846,
            "unit": "iter/sec",
            "range": "stddev: 0.001328255814714612",
            "extra": "mean: 52.636419300000625 msec\nrounds: 20"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1425.9563141244196,
            "unit": "iter/sec",
            "range": "stddev: 0.00002188658025986457",
            "extra": "mean: 701.2837560974161 usec\nrounds: 1189"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 17.84320207665003,
            "unit": "iter/sec",
            "range": "stddev: 0.0011403873349016753",
            "extra": "mean: 56.04375244444605 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1424.742947189324,
            "unit": "iter/sec",
            "range": "stddev: 0.000034435024138906414",
            "extra": "mean: 701.8809968301721 usec\nrounds: 1262"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 17.45091089304975,
            "unit": "iter/sec",
            "range": "stddev: 0.0011991188880202555",
            "extra": "mean: 57.303598999996865 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1438.8923005601746,
            "unit": "iter/sec",
            "range": "stddev: 0.000023238456283618124",
            "extra": "mean: 694.9790471536267 usec\nrounds: 1230"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 17.967212092344873,
            "unit": "iter/sec",
            "range": "stddev: 0.0010563240190419694",
            "extra": "mean: 55.656937473680784 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1452.0466008336043,
            "unit": "iter/sec",
            "range": "stddev: 0.000020601106199217282",
            "extra": "mean: 688.6831313994404 usec\nrounds: 1309"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 18.960607706878072,
            "unit": "iter/sec",
            "range": "stddev: 0.0011069379830112652",
            "extra": "mean: 52.74092557894355 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1427.632435490084,
            "unit": "iter/sec",
            "range": "stddev: 0.000022115273534238",
            "extra": "mean: 700.4604092346191 usec\nrounds: 1256"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 17.340721363082572,
            "unit": "iter/sec",
            "range": "stddev: 0.0010658466074203448",
            "extra": "mean: 57.667727833338255 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1436.2640001291345,
            "unit": "iter/sec",
            "range": "stddev: 0.000020718193403409736",
            "extra": "mean: 696.2508284758861 usec\nrounds: 1201"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 17.894895209833646,
            "unit": "iter/sec",
            "range": "stddev: 0.00031115950530122435",
            "extra": "mean: 55.8818583888928 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1442.9237581179639,
            "unit": "iter/sec",
            "range": "stddev: 0.00003839900654761107",
            "extra": "mean: 693.0373100962183 usec\nrounds: 1248"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 18.981363873917232,
            "unit": "iter/sec",
            "range": "stddev: 0.0012260234323935252",
            "extra": "mean: 52.68325325000092 msec\nrounds: 20"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "akoumjian@users.noreply.github.com",
            "name": "Alec Koumjian",
            "username": "akoumjian"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "7eeb381fb8a66c31599021501134ebe30f2740e4",
          "message": "Fixes bug where the aberrated coordinates were misaligned with ephemeris produced (#132)",
          "timestamp": "2024-12-11T10:03:53-05:00",
          "tree_id": "a28ae499b6eda19074ca169de171be5b7b472cb4",
          "url": "https://github.com/B612-Asteroid-Institute/adam_core/commit/7eeb381fb8a66c31599021501134ebe30f2740e4"
        },
        "date": 1733929680813,
        "tool": "pytest",
        "benches": [
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 443.1259267659649,
            "unit": "iter/sec",
            "range": "stddev: 0.00007652021347583382",
            "extra": "mean: 2.2566948571441765 msec\nrounds: 7"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 303.75616981919217,
            "unit": "iter/sec",
            "range": "stddev: 0.00020512637953159466",
            "extra": "mean: 3.292114200002061 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 328.10332262339773,
            "unit": "iter/sec",
            "range": "stddev: 0.00015940270097626635",
            "extra": "mean: 3.0478204000019105 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 439.5821718370692,
            "unit": "iter/sec",
            "range": "stddev: 0.000037034419121317884",
            "extra": "mean: 2.274887527446516 msec\nrounds: 419"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 306.2697432961003,
            "unit": "iter/sec",
            "range": "stddev: 0.00003700328413246498",
            "extra": "mean: 3.265095628572112 msec\nrounds: 280"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 333.18440235022155,
            "unit": "iter/sec",
            "range": "stddev: 0.00003211295515294028",
            "extra": "mean: 3.0013409779875158 msec\nrounds: 318"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 419.6995651389698,
            "unit": "iter/sec",
            "range": "stddev: 0.00003320090505913895",
            "extra": "mean: 2.382656745591058 msec\nrounds: 397"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 261.5175025868041,
            "unit": "iter/sec",
            "range": "stddev: 0.00007895078266374757",
            "extra": "mean: 3.8238358431404627 msec\nrounds: 51"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 268.6011969517998,
            "unit": "iter/sec",
            "range": "stddev: 0.00015410964202424387",
            "extra": "mean: 3.7229915999944296 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 422.7994438434323,
            "unit": "iter/sec",
            "range": "stddev: 0.00011388173922920997",
            "extra": "mean: 2.365187595588021 msec\nrounds: 408"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 261.0392969107388,
            "unit": "iter/sec",
            "range": "stddev: 0.00013260638228300705",
            "extra": "mean: 3.8308408421048785 msec\nrounds: 247"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 275.0044931423791,
            "unit": "iter/sec",
            "range": "stddev: 0.00013511277498840825",
            "extra": "mean: 3.6363042238814125 msec\nrounds: 268"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 413.45458330848584,
            "unit": "iter/sec",
            "range": "stddev: 0.000026670847399081136",
            "extra": "mean: 2.418645337047533 msec\nrounds: 359"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 229.05500740875806,
            "unit": "iter/sec",
            "range": "stddev: 0.00006911169176791839",
            "extra": "mean: 4.365763540001808 msec\nrounds: 50"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 234.38738590892828,
            "unit": "iter/sec",
            "range": "stddev: 0.0001231454007025004",
            "extra": "mean: 4.2664411999908225 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 392.0425278484931,
            "unit": "iter/sec",
            "range": "stddev: 0.0002583434161278829",
            "extra": "mean: 2.550743679487893 msec\nrounds: 390"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 225.68168357036933,
            "unit": "iter/sec",
            "range": "stddev: 0.00009115067002184266",
            "extra": "mean: 4.4310197627898855 msec\nrounds: 215"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 230.655802998441,
            "unit": "iter/sec",
            "range": "stddev: 0.00004853641180021903",
            "extra": "mean: 4.335464302221605 msec\nrounds: 225"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_to_matrix",
            "value": 647.0650587196996,
            "unit": "iter/sec",
            "range": "stddev: 0.00010772300588912747",
            "extra": "mean: 1.5454396532840562 msec\nrounds: 548"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_from_matrix",
            "value": 10958.034696204166,
            "unit": "iter/sec",
            "range": "stddev: 0.000006373591839184419",
            "extra": "mean: 91.25723979924953 usec\nrounds: 2794"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body",
            "value": 44748.29086620716,
            "unit": "iter/sec",
            "range": "stddev: 0.000009343175081224558",
            "extra": "mean: 22.347222221065344 usec\nrounds: 9"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body_vmap",
            "value": 33992.4083643204,
            "unit": "iter/sec",
            "range": "stddev: 0.000010755189148838043",
            "extra": "mean: 29.4183333314398 usec\nrounds: 6"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body",
            "value": 343.26509871552486,
            "unit": "iter/sec",
            "range": "stddev: 0.00007237792880021504",
            "extra": "mean: 2.913200333334013 msec\nrounds: 6"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body_matrix",
            "value": 3.9373896240919057,
            "unit": "iter/sec",
            "range": "stddev: 0.0026348963614046005",
            "extra": "mean: 253.97537339999303 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=1,]",
            "value": 577.9120693654526,
            "unit": "iter/sec",
            "range": "stddev: 0.000026022567781215662",
            "extra": "mean: 1.7303670454538178 msec\nrounds: 110"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.7448405529071245,
            "unit": "iter/sec",
            "range": "stddev: 0.001947518445778777",
            "extra": "mean: 210.75523800000155 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=1,]",
            "value": 1179.998944569196,
            "unit": "iter/sec",
            "range": "stddev: 0.00001785066859045412",
            "extra": "mean: 847.4583851132921 usec\nrounds: 927"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=10000,]",
            "value": 17.184001698237715,
            "unit": "iter/sec",
            "range": "stddev: 0.0013465051227030897",
            "extra": "mean: 58.19366277777741 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 568.2227375405881,
            "unit": "iter/sec",
            "range": "stddev: 0.00007314537497714012",
            "extra": "mean: 1.7598732573220373 msec\nrounds: 478"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.7427972963436895,
            "unit": "iter/sec",
            "range": "stddev: 0.0019582440673063975",
            "extra": "mean: 210.8460340000022 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1171.699446058479,
            "unit": "iter/sec",
            "range": "stddev: 0.000018200860845367336",
            "extra": "mean: 853.4611869655953 usec\nrounds: 936"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 17.189442020484282,
            "unit": "iter/sec",
            "range": "stddev: 0.0011099958334278642",
            "extra": "mean: 58.17524494444449 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=1,]",
            "value": 553.1953321345782,
            "unit": "iter/sec",
            "range": "stddev: 0.00027624752314278216",
            "extra": "mean: 1.8076797505527864 msec\nrounds: 453"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.770275045405553,
            "unit": "iter/sec",
            "range": "stddev: 0.0017742618384931192",
            "extra": "mean: 209.63151820001258 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=1,]",
            "value": 1178.1314933008496,
            "unit": "iter/sec",
            "range": "stddev: 0.000020873808144564023",
            "extra": "mean: 848.8016878304758 usec\nrounds: 945"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=10000,]",
            "value": 17.509796439277864,
            "unit": "iter/sec",
            "range": "stddev: 0.0013174505893896074",
            "extra": "mean: 57.11088666666657 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 574.5978974679099,
            "unit": "iter/sec",
            "range": "stddev: 0.00003294801693064504",
            "extra": "mean: 1.740347475002461 msec\nrounds: 480"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.74933818043673,
            "unit": "iter/sec",
            "range": "stddev: 0.0022122861937824796",
            "extra": "mean: 210.55565260001003 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1177.5266846112024,
            "unit": "iter/sec",
            "range": "stddev: 0.00003526540068834842",
            "extra": "mean: 849.2376547119877 usec\nrounds: 976"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 17.449758273052634,
            "unit": "iter/sec",
            "range": "stddev: 0.0011340541638296734",
            "extra": "mean: 57.307384111118786 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/orbits/tests/test_benchmarks.py::test_benchmark_iterate_real_orbits",
            "value": 19275.551911255458,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023554349608279198",
            "extra": "mean: 51.87918896454923 usec\nrounds: 14934"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1439.5028583360659,
            "unit": "iter/sec",
            "range": "stddev: 0.000035448845876765784",
            "extra": "mean: 694.6842753448291 usec\nrounds: 868"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 16.904965287146677,
            "unit": "iter/sec",
            "range": "stddev: 0.004559903290366061",
            "extra": "mean: 59.15421788889022 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1465.9042250787422,
            "unit": "iter/sec",
            "range": "stddev: 0.00003494155221538919",
            "extra": "mean: 682.1728069896819 usec\nrounds: 1259"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 19.439426577659034,
            "unit": "iter/sec",
            "range": "stddev: 0.00014100021727286556",
            "extra": "mean: 51.44184660000519 msec\nrounds: 20"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1447.7567875750597,
            "unit": "iter/sec",
            "range": "stddev: 0.000041682982266900554",
            "extra": "mean: 690.7237517946394 usec\nrounds: 1253"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 18.269761804451228,
            "unit": "iter/sec",
            "range": "stddev: 0.00020804707815559034",
            "extra": "mean: 54.73525110526405 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1448.8289856985164,
            "unit": "iter/sec",
            "range": "stddev: 0.000030852062670648004",
            "extra": "mean: 690.2125853852069 usec\nrounds: 1218"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 17.238312426328807,
            "unit": "iter/sec",
            "range": "stddev: 0.0005640364113604417",
            "extra": "mean: 58.01031883333647 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1459.2716827772495,
            "unit": "iter/sec",
            "range": "stddev: 0.000025193821437558278",
            "extra": "mean: 685.2733536888929 usec\nrounds: 1261"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 19.413842738484394,
            "unit": "iter/sec",
            "range": "stddev: 0.001101284020515789",
            "extra": "mean: 51.50963740000236 msec\nrounds: 20"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1452.3714745044288,
            "unit": "iter/sec",
            "range": "stddev: 0.000030931504347804296",
            "extra": "mean: 688.5290833333222 usec\nrounds: 1224"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 18.064866495429893,
            "unit": "iter/sec",
            "range": "stddev: 0.001128381916842499",
            "extra": "mean: 55.35606921052992 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1438.6518939334592,
            "unit": "iter/sec",
            "range": "stddev: 0.00003724796380346633",
            "extra": "mean: 695.0951819664112 usec\nrounds: 1231"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 17.739770166888057,
            "unit": "iter/sec",
            "range": "stddev: 0.0002286358295154946",
            "extra": "mean: 56.37051611111272 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1452.8551282905253,
            "unit": "iter/sec",
            "range": "stddev: 0.00003001383501446023",
            "extra": "mean: 688.2998728005532 usec\nrounds: 1250"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 18.302572935923262,
            "unit": "iter/sec",
            "range": "stddev: 0.00015998367452988306",
            "extra": "mean: 54.637126894725064 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1452.2222733068832,
            "unit": "iter/sec",
            "range": "stddev: 0.000016705713506876622",
            "extra": "mean: 688.5998227550118 usec\nrounds: 1292"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 19.47569239221991,
            "unit": "iter/sec",
            "range": "stddev: 0.00017575897568078105",
            "extra": "mean: 51.346056400001316 msec\nrounds: 20"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1459.6228968108405,
            "unit": "iter/sec",
            "range": "stddev: 0.000015868688108611565",
            "extra": "mean: 685.1084634153932 usec\nrounds: 1189"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 17.67574685526939,
            "unit": "iter/sec",
            "range": "stddev: 0.00020550523693711823",
            "extra": "mean: 56.57469572222832 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1457.7376210547743,
            "unit": "iter/sec",
            "range": "stddev: 0.000039095716787712414",
            "extra": "mean: 685.9945065260994 usec\nrounds: 1226"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 18.06868291237998,
            "unit": "iter/sec",
            "range": "stddev: 0.00010788303008821214",
            "extra": "mean: 55.34437705555383 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1470.7475830480075,
            "unit": "iter/sec",
            "range": "stddev: 0.000016903722306197852",
            "extra": "mean: 679.9263255817015 usec\nrounds: 1290"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 19.417565444847526,
            "unit": "iter/sec",
            "range": "stddev: 0.00016249856033810753",
            "extra": "mean: 51.49976205000257 msec\nrounds: 20"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "akoumjian@users.noreply.github.com",
            "name": "Alec Koumjian",
            "username": "akoumjian"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "7eeb381fb8a66c31599021501134ebe30f2740e4",
          "message": "Fixes bug where the aberrated coordinates were misaligned with ephemeris produced (#132)",
          "timestamp": "2024-12-11T10:03:53-05:00",
          "tree_id": "a28ae499b6eda19074ca169de171be5b7b472cb4",
          "url": "https://github.com/B612-Asteroid-Institute/adam_core/commit/7eeb381fb8a66c31599021501134ebe30f2740e4"
        },
        "date": 1733929685440,
        "tool": "pytest",
        "benches": [
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 420.80890693497355,
            "unit": "iter/sec",
            "range": "stddev: 0.00006358720745302837",
            "extra": "mean: 2.3763755555547856 msec\nrounds: 9"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 278.260374019393,
            "unit": "iter/sec",
            "range": "stddev: 0.00019792770325740758",
            "extra": "mean: 3.593756400005077 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 300.20872311575425,
            "unit": "iter/sec",
            "range": "stddev: 0.00015282701033083798",
            "extra": "mean: 3.3310158000119827 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 415.4443091989067,
            "unit": "iter/sec",
            "range": "stddev: 0.00007874197530613903",
            "extra": "mean: 2.4070614950251237 msec\nrounds: 402"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 290.1393599530919,
            "unit": "iter/sec",
            "range": "stddev: 0.00010366515592835385",
            "extra": "mean: 3.4466195836430957 msec\nrounds: 269"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 310.4768750294959,
            "unit": "iter/sec",
            "range": "stddev: 0.0001026010001164825",
            "extra": "mean: 3.2208517942117205 msec\nrounds: 311"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 395.8221752055867,
            "unit": "iter/sec",
            "range": "stddev: 0.0001184651476186753",
            "extra": "mean: 2.52638700568155 msec\nrounds: 352"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 250.45997349815977,
            "unit": "iter/sec",
            "range": "stddev: 0.0000742115238598104",
            "extra": "mean: 3.992653940001105 msec\nrounds: 50"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 257.57096363349393,
            "unit": "iter/sec",
            "range": "stddev: 0.00017114367061071926",
            "extra": "mean: 3.882425200004036 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 409.86087129744647,
            "unit": "iter/sec",
            "range": "stddev: 0.00005904795741687092",
            "extra": "mean: 2.4398523255816595 msec\nrounds: 387"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 250.86034744512253,
            "unit": "iter/sec",
            "range": "stddev: 0.000038638187405517095",
            "extra": "mean: 3.9862816510638734 msec\nrounds: 235"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 257.1792607146896,
            "unit": "iter/sec",
            "range": "stddev: 0.00013063907802329707",
            "extra": "mean: 3.8883384189729964 msec\nrounds: 253"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 387.2614040664166,
            "unit": "iter/sec",
            "range": "stddev: 0.000050198811235110333",
            "extra": "mean: 2.5822351246459267 msec\nrounds: 353"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 219.01253603528127,
            "unit": "iter/sec",
            "range": "stddev: 0.00011965610994272872",
            "extra": "mean: 4.565948680850431 msec\nrounds: 47"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 220.99353744165944,
            "unit": "iter/sec",
            "range": "stddev: 0.0001286688228942931",
            "extra": "mean: 4.525019200002589 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 382.15769760419863,
            "unit": "iter/sec",
            "range": "stddev: 0.00009356998709302796",
            "extra": "mean: 2.6167208099408787 msec\nrounds: 342"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 218.71335884268674,
            "unit": "iter/sec",
            "range": "stddev: 0.00004518443127245431",
            "extra": "mean: 4.5721944251209035 msec\nrounds: 207"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 224.93886942282916,
            "unit": "iter/sec",
            "range": "stddev: 0.00013579462412415352",
            "extra": "mean: 4.445652290179553 msec\nrounds: 224"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_to_matrix",
            "value": 600.0988664542267,
            "unit": "iter/sec",
            "range": "stddev: 0.00001979305908611697",
            "extra": "mean: 1.6663920828723584 msec\nrounds: 543"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_from_matrix",
            "value": 10664.190459259302,
            "unit": "iter/sec",
            "range": "stddev: 0.000005834032749149991",
            "extra": "mean: 93.77176859512471 usec\nrounds: 2904"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body",
            "value": 44552.69097577035,
            "unit": "iter/sec",
            "range": "stddev: 0.000008671608328198454",
            "extra": "mean: 22.44533333674149 usec\nrounds: 9"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body_vmap",
            "value": 34369.491684620254,
            "unit": "iter/sec",
            "range": "stddev: 0.000010330398429020455",
            "extra": "mean: 29.095571420612032 usec\nrounds: 7"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body",
            "value": 331.6103082780648,
            "unit": "iter/sec",
            "range": "stddev: 0.0000844908939587076",
            "extra": "mean: 3.0155878000073244 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body_matrix",
            "value": 3.982911207940811,
            "unit": "iter/sec",
            "range": "stddev: 0.0018029510918879305",
            "extra": "mean: 251.07263200000034 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=1,]",
            "value": 543.9137833735767,
            "unit": "iter/sec",
            "range": "stddev: 0.00003065124944078451",
            "extra": "mean: 1.8385266756756726 msec\nrounds: 111"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.456966702435871,
            "unit": "iter/sec",
            "range": "stddev: 0.00047462864572338405",
            "extra": "mean: 224.36784180000018 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=1,]",
            "value": 1114.4814500660598,
            "unit": "iter/sec",
            "range": "stddev: 0.000027727282149366415",
            "extra": "mean: 897.2782812497471 usec\nrounds: 928"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=10000,]",
            "value": 16.329739835439256,
            "unit": "iter/sec",
            "range": "stddev: 0.00017067035941183018",
            "extra": "mean: 61.23796276470811 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 545.6676356269126,
            "unit": "iter/sec",
            "range": "stddev: 0.000023715966510908046",
            "extra": "mean: 1.8326173932802685 msec\nrounds: 506"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.433495330681391,
            "unit": "iter/sec",
            "range": "stddev: 0.0020995580580848885",
            "extra": "mean: 225.5556677999948 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1121.203626925575,
            "unit": "iter/sec",
            "range": "stddev: 0.0000181288598209117",
            "extra": "mean: 891.8986489029434 usec\nrounds: 957"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 16.28419586299817,
            "unit": "iter/sec",
            "range": "stddev: 0.00017509007138600435",
            "extra": "mean: 61.40923435293811 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=1,]",
            "value": 546.6008469761498,
            "unit": "iter/sec",
            "range": "stddev: 0.000024873414913124028",
            "extra": "mean: 1.8294885665327805 msec\nrounds: 496"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.483981997558432,
            "unit": "iter/sec",
            "range": "stddev: 0.00028877694266739675",
            "extra": "mean: 223.0160603999991 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=1,]",
            "value": 1119.5766000116873,
            "unit": "iter/sec",
            "range": "stddev: 0.0000350451384390035",
            "extra": "mean: 893.1948023829374 usec\nrounds: 1007"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=10000,]",
            "value": 16.606263028229385,
            "unit": "iter/sec",
            "range": "stddev: 0.00028540645491984597",
            "extra": "mean: 60.21824406250076 msec\nrounds: 16"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 547.2257872497149,
            "unit": "iter/sec",
            "range": "stddev: 0.000025598751321002264",
            "extra": "mean: 1.8273992624248738 msec\nrounds: 503"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.4566985312261025,
            "unit": "iter/sec",
            "range": "stddev: 0.000552380176473245",
            "extra": "mean: 224.3813426000088 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1117.1138490315593,
            "unit": "iter/sec",
            "range": "stddev: 0.00004075410912649779",
            "extra": "mean: 895.1639090920886 usec\nrounds: 946"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 16.498053662985235,
            "unit": "iter/sec",
            "range": "stddev: 0.000247331562130689",
            "extra": "mean: 60.61321052940831 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/orbits/tests/test_benchmarks.py::test_benchmark_iterate_real_orbits",
            "value": 17102.912013440808,
            "unit": "iter/sec",
            "range": "stddev: 0.000004486298535503477",
            "extra": "mean: 58.46957519363495 usec\nrounds: 14456"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1356.8914047637643,
            "unit": "iter/sec",
            "range": "stddev: 0.00001774369103112248",
            "extra": "mean: 736.978653184188 usec\nrounds: 989"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 16.29771654677894,
            "unit": "iter/sec",
            "range": "stddev: 0.0006923515001173184",
            "extra": "mean: 61.35828888235504 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1372.116436016319,
            "unit": "iter/sec",
            "range": "stddev: 0.00001674870352332005",
            "extra": "mean: 728.801123396868 usec\nrounds: 1248"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 18.065540008728842,
            "unit": "iter/sec",
            "range": "stddev: 0.00035206191970549365",
            "extra": "mean: 55.35400544444416 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1353.001495496747,
            "unit": "iter/sec",
            "range": "stddev: 0.00002510142473727441",
            "extra": "mean: 739.0974831353424 usec\nrounds: 1186"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 17.08472060473572,
            "unit": "iter/sec",
            "range": "stddev: 0.0009597927248591548",
            "extra": "mean: 58.531832222226086 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1363.0457610938342,
            "unit": "iter/sec",
            "range": "stddev: 0.000030212612032717287",
            "extra": "mean: 733.6510838766758 usec\nrounds: 1228"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 16.39514578927004,
            "unit": "iter/sec",
            "range": "stddev: 0.0002158577516900278",
            "extra": "mean: 60.99366317647871 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1380.8055885191807,
            "unit": "iter/sec",
            "range": "stddev: 0.000029377046915443183",
            "extra": "mean: 724.2149136088242 usec\nrounds: 1308"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 18.056251325896977,
            "unit": "iter/sec",
            "range": "stddev: 0.0002234209347616478",
            "extra": "mean: 55.382481222210345 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1369.1654122482005,
            "unit": "iter/sec",
            "range": "stddev: 0.00003097074749545681",
            "extra": "mean: 730.371941223652 usec\nrounds: 1225"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 17.08528868474006,
            "unit": "iter/sec",
            "range": "stddev: 0.00021401611017256037",
            "extra": "mean: 58.52988605882688 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1367.1685339064252,
            "unit": "iter/sec",
            "range": "stddev: 0.000030748551665944064",
            "extra": "mean: 731.4387181971556 usec\nrounds: 1242"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 16.689913966017706,
            "unit": "iter/sec",
            "range": "stddev: 0.0005728872041481213",
            "extra": "mean: 59.916426294113776 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1367.686298204968,
            "unit": "iter/sec",
            "range": "stddev: 0.000016052995886906898",
            "extra": "mean: 731.1618178177694 usec\nrounds: 1246"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 17.1713551336369,
            "unit": "iter/sec",
            "range": "stddev: 0.000246529976029229",
            "extra": "mean: 58.23652194118936 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1374.0069055083466,
            "unit": "iter/sec",
            "range": "stddev: 0.00002992194270574956",
            "extra": "mean: 727.798380045278 usec\nrounds: 1313"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 17.999141638933637,
            "unit": "iter/sec",
            "range": "stddev: 0.0004940678911616956",
            "extra": "mean: 55.55820494444674 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1362.1865872653225,
            "unit": "iter/sec",
            "range": "stddev: 0.000031131410292935305",
            "extra": "mean: 734.1138206385989 usec\nrounds: 1221"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 16.60004090523352,
            "unit": "iter/sec",
            "range": "stddev: 0.0002679946521672141",
            "extra": "mean: 60.24081541176977 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1362.0671265468152,
            "unit": "iter/sec",
            "range": "stddev: 0.00003155003556317494",
            "extra": "mean: 734.178206426032 usec\nrounds: 1245"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 17.10005744643457,
            "unit": "iter/sec",
            "range": "stddev: 0.00024228763556152395",
            "extra": "mean: 58.479335705886996 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1371.0044961086087,
            "unit": "iter/sec",
            "range": "stddev: 0.000029433515358982066",
            "extra": "mean: 729.3922104838829 usec\nrounds: 1259"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 17.762202370046946,
            "unit": "iter/sec",
            "range": "stddev: 0.0022869442703491454",
            "extra": "mean: 56.29932477778413 msec\nrounds: 18"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "akoumjian@users.noreply.github.com",
            "name": "Alec Koumjian",
            "username": "akoumjian"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "9e21f2c06dbe80ed0e2a195c595a17e5d0998854",
          "message": "Nt/seed propagate (#135)\n\n* add a seed for variant orbits in propagate_orbits\n\n* Ensure seed is passed to all VariantOrbits.create and sort varianats before collapse",
          "timestamp": "2025-01-10T09:19:58-05:00",
          "tree_id": "6bafeb72e1fe65ad23422c9206b7f95bdfcd192e",
          "url": "https://github.com/B612-Asteroid-Institute/adam_core/commit/9e21f2c06dbe80ed0e2a195c595a17e5d0998854"
        },
        "date": 1736519067839,
        "tool": "pytest",
        "benches": [
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 421.7076545407109,
            "unit": "iter/sec",
            "range": "stddev: 0.00008266279369025892",
            "extra": "mean: 2.3713110000080917 msec\nrounds: 8"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 294.3347102513538,
            "unit": "iter/sec",
            "range": "stddev: 0.00022279796875660556",
            "extra": "mean: 3.397492599992802 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 319.6686085043398,
            "unit": "iter/sec",
            "range": "stddev: 0.0002007939531685233",
            "extra": "mean: 3.1282395999994606 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 443.87525826842614,
            "unit": "iter/sec",
            "range": "stddev: 0.00013610822587745204",
            "extra": "mean: 2.252885200001994 msec\nrounds: 420"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 309.9000156068121,
            "unit": "iter/sec",
            "range": "stddev: 0.00006217002710059901",
            "extra": "mean: 3.2268472076127845 msec\nrounds: 289"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 330.5030041409851,
            "unit": "iter/sec",
            "range": "stddev: 0.000045700184968184134",
            "extra": "mean: 3.025691105589535 msec\nrounds: 322"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 419.9976586069477,
            "unit": "iter/sec",
            "range": "stddev: 0.00004605388058645321",
            "extra": "mean: 2.38096565422962 msec\nrounds: 402"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 261.25346276395885,
            "unit": "iter/sec",
            "range": "stddev: 0.00007291460209193662",
            "extra": "mean: 3.827700461537977 msec\nrounds: 52"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 263.57764907891675,
            "unit": "iter/sec",
            "range": "stddev: 0.00018261370659260458",
            "extra": "mean: 3.793948400004865 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 425.95307818728907,
            "unit": "iter/sec",
            "range": "stddev: 0.00004742408330128845",
            "extra": "mean: 2.347676425430845 msec\nrounds: 409"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 261.62235832682785,
            "unit": "iter/sec",
            "range": "stddev: 0.000052630757088411835",
            "extra": "mean: 3.8223032862916284 msec\nrounds: 248"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 269.4696719574798,
            "unit": "iter/sec",
            "range": "stddev: 0.00003820797894697568",
            "extra": "mean: 3.710992753788604 msec\nrounds: 264"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 409.30696508788094,
            "unit": "iter/sec",
            "range": "stddev: 0.00005230761268488621",
            "extra": "mean: 2.4431541246440633 msec\nrounds: 353"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 227.8829834822323,
            "unit": "iter/sec",
            "range": "stddev: 0.0001127202002264629",
            "extra": "mean: 4.3882170784286245 msec\nrounds: 51"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 233.3977417740894,
            "unit": "iter/sec",
            "range": "stddev: 0.00020698203078133907",
            "extra": "mean: 4.2845316000011735 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 411.960459561351,
            "unit": "iter/sec",
            "range": "stddev: 0.00008141562667058086",
            "extra": "mean: 2.427417429975644 msec\nrounds: 407"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 229.07656615912262,
            "unit": "iter/sec",
            "range": "stddev: 0.00010932787754121461",
            "extra": "mean: 4.365352671234707 msec\nrounds: 219"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 234.0698973624243,
            "unit": "iter/sec",
            "range": "stddev: 0.00008245631297587205",
            "extra": "mean: 4.27222813043593 msec\nrounds: 207"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_to_matrix",
            "value": 655.8067339660496,
            "unit": "iter/sec",
            "range": "stddev: 0.00011443516316549964",
            "extra": "mean: 1.5248394812179056 msec\nrounds: 559"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_from_matrix",
            "value": 11012.492153613191,
            "unit": "iter/sec",
            "range": "stddev: 0.0000072128179666779965",
            "extra": "mean: 90.80596708274616 usec\nrounds: 2886"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body",
            "value": 45141.51863826487,
            "unit": "iter/sec",
            "range": "stddev: 0.000008584190770466247",
            "extra": "mean: 22.152555566713595 usec\nrounds: 9"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body_vmap",
            "value": 35189.849238052266,
            "unit": "iter/sec",
            "range": "stddev: 0.000010020118683428628",
            "extra": "mean: 28.41728571313849 usec\nrounds: 7"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body",
            "value": 319.80584800224386,
            "unit": "iter/sec",
            "range": "stddev: 0.0003176017666514002",
            "extra": "mean: 3.1268971666615166 msec\nrounds: 6"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body_matrix",
            "value": 3.914600485073544,
            "unit": "iter/sec",
            "range": "stddev: 0.0008351114874355493",
            "extra": "mean: 255.453910000017 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=1,]",
            "value": 565.6720722842465,
            "unit": "iter/sec",
            "range": "stddev: 0.00006251027905576291",
            "extra": "mean: 1.767808681029433 msec\nrounds: 116"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.738929663640391,
            "unit": "iter/sec",
            "range": "stddev: 0.0012497058655570694",
            "extra": "mean: 211.0181139999895 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=1,]",
            "value": 1164.0783477157108,
            "unit": "iter/sec",
            "range": "stddev: 0.00002188623252100915",
            "extra": "mean: 859.0487074708637 usec\nrounds: 964"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=10000,]",
            "value": 17.413542619481134,
            "unit": "iter/sec",
            "range": "stddev: 0.00020003264455042724",
            "extra": "mean: 57.42656861110302 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 568.6085823074976,
            "unit": "iter/sec",
            "range": "stddev: 0.00003991175661630989",
            "extra": "mean: 1.7586790476180507 msec\nrounds: 504"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.743666187648663,
            "unit": "iter/sec",
            "range": "stddev: 0.00022595051228212592",
            "extra": "mean: 210.80741360000275 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1158.8907397687217,
            "unit": "iter/sec",
            "range": "stddev: 0.000038626346063900456",
            "extra": "mean: 862.8941156260931 usec\nrounds: 960"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 17.24126704393359,
            "unit": "iter/sec",
            "range": "stddev: 0.00032546923205666175",
            "extra": "mean: 58.00037766666656 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=1,]",
            "value": 570.3221612493546,
            "unit": "iter/sec",
            "range": "stddev: 0.000030157620708607058",
            "extra": "mean: 1.7533949545453185 msec\nrounds: 506"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.778142350358226,
            "unit": "iter/sec",
            "range": "stddev: 0.0007350079466280989",
            "extra": "mean: 209.2863557999749 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=1,]",
            "value": 1170.602648190551,
            "unit": "iter/sec",
            "range": "stddev: 0.000021118212664573448",
            "extra": "mean: 854.2608386763361 usec\nrounds: 967"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=10000,]",
            "value": 17.56751338128652,
            "unit": "iter/sec",
            "range": "stddev: 0.0010525003470635442",
            "extra": "mean: 56.92325249999423 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 569.873577979027,
            "unit": "iter/sec",
            "range": "stddev: 0.00003636253262880087",
            "extra": "mean: 1.7547751617935212 msec\nrounds: 513"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.755806350657637,
            "unit": "iter/sec",
            "range": "stddev: 0.0008345918398945289",
            "extra": "mean: 210.26928479998332 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1161.71192765359,
            "unit": "iter/sec",
            "range": "stddev: 0.000030270789410918917",
            "extra": "mean: 860.7985992015994 usec\nrounds: 1003"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 17.46635777658777,
            "unit": "iter/sec",
            "range": "stddev: 0.0004259044142988797",
            "extra": "mean: 57.25292088888838 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/orbits/tests/test_benchmarks.py::test_benchmark_iterate_real_orbits",
            "value": 19543.478279901617,
            "unit": "iter/sec",
            "range": "stddev: 0.0000029104602294227084",
            "extra": "mean: 51.16796435506535 usec\nrounds: 15907"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1405.5649112471585,
            "unit": "iter/sec",
            "range": "stddev: 0.000023681470012640244",
            "extra": "mean: 711.4577149714839 usec\nrounds: 1042"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 17.239193730647486,
            "unit": "iter/sec",
            "range": "stddev: 0.0011210294256888713",
            "extra": "mean: 58.007353222222946 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1436.4921226469594,
            "unit": "iter/sec",
            "range": "stddev: 0.000017507941298964098",
            "extra": "mean: 696.1402601758408 usec\nrounds: 1253"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 19.070618117474748,
            "unit": "iter/sec",
            "range": "stddev: 0.00029417184680536143",
            "extra": "mean: 52.436685263163135 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1427.9820190927735,
            "unit": "iter/sec",
            "range": "stddev: 0.000021012848623169618",
            "extra": "mean: 700.288929853137 usec\nrounds: 1226"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 18.150774573364462,
            "unit": "iter/sec",
            "range": "stddev: 0.0007779360003052797",
            "extra": "mean: 55.09406752632255 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1423.2960518138516,
            "unit": "iter/sec",
            "range": "stddev: 0.000024523419993434984",
            "extra": "mean: 702.5945155441117 usec\nrounds: 1158"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 17.21894989631478,
            "unit": "iter/sec",
            "range": "stddev: 0.00026201367599153365",
            "extra": "mean: 58.0755508333305 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1432.479014009145,
            "unit": "iter/sec",
            "range": "stddev: 0.000023417483685760287",
            "extra": "mean: 698.0905061926554 usec\nrounds: 1292"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 19.025908565233323,
            "unit": "iter/sec",
            "range": "stddev: 0.00027097818841962306",
            "extra": "mean: 52.55990780000559 msec\nrounds: 20"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1423.1652917880263,
            "unit": "iter/sec",
            "range": "stddev: 0.000023232973858048664",
            "extra": "mean: 702.6590697301416 usec\nrounds: 1219"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 18.022043265736258,
            "unit": "iter/sec",
            "range": "stddev: 0.00024080327648948123",
            "extra": "mean: 55.487603999997766 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1422.2808715853123,
            "unit": "iter/sec",
            "range": "stddev: 0.000023673726194312348",
            "extra": "mean: 703.0960058440309 usec\nrounds: 1198"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 17.50553317810221,
            "unit": "iter/sec",
            "range": "stddev: 0.00046959279775041164",
            "extra": "mean: 57.124795333335335 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1431.094893087702,
            "unit": "iter/sec",
            "range": "stddev: 0.00002578006898701678",
            "extra": "mean: 698.7656827161334 usec\nrounds: 1267"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 18.026201759768817,
            "unit": "iter/sec",
            "range": "stddev: 0.0004802699417588725",
            "extra": "mean: 55.47480347367558 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1435.1229744622908,
            "unit": "iter/sec",
            "range": "stddev: 0.000020446788559319686",
            "extra": "mean: 696.8043978075664 usec\nrounds: 1277"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 18.820941262207064,
            "unit": "iter/sec",
            "range": "stddev: 0.0013790843783098444",
            "extra": "mean: 53.13230544999499 msec\nrounds: 20"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1416.5477687610614,
            "unit": "iter/sec",
            "range": "stddev: 0.00003814074228605604",
            "extra": "mean: 705.9416011608407 usec\nrounds: 1206"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 17.354250022663564,
            "unit": "iter/sec",
            "range": "stddev: 0.00044772322443259634",
            "extra": "mean: 57.62277244444805 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1418.9823061472591,
            "unit": "iter/sec",
            "range": "stddev: 0.00002031212424215221",
            "extra": "mean: 704.7304224075518 usec\nrounds: 1205"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 17.757958019985,
            "unit": "iter/sec",
            "range": "stddev: 0.0008931010863216133",
            "extra": "mean: 56.3127809444413 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1429.9675902092276,
            "unit": "iter/sec",
            "range": "stddev: 0.000023059181332257106",
            "extra": "mean: 699.3165487433765 usec\nrounds: 1272"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 18.909268448557338,
            "unit": "iter/sec",
            "range": "stddev: 0.0002598759407282047",
            "extra": "mean: 52.88411884999675 msec\nrounds: 20"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "akoumjian@users.noreply.github.com",
            "name": "Alec Koumjian",
            "username": "akoumjian"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "9e21f2c06dbe80ed0e2a195c595a17e5d0998854",
          "message": "Nt/seed propagate (#135)\n\n* add a seed for variant orbits in propagate_orbits\n\n* Ensure seed is passed to all VariantOrbits.create and sort varianats before collapse",
          "timestamp": "2025-01-10T09:19:58-05:00",
          "tree_id": "6bafeb72e1fe65ad23422c9206b7f95bdfcd192e",
          "url": "https://github.com/B612-Asteroid-Institute/adam_core/commit/9e21f2c06dbe80ed0e2a195c595a17e5d0998854"
        },
        "date": 1736519080544,
        "tool": "pytest",
        "benches": [
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 403.20036255561865,
            "unit": "iter/sec",
            "range": "stddev: 0.00009303554480421863",
            "extra": "mean: 2.4801565000132086 msec\nrounds: 8"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 272.8708325536278,
            "unit": "iter/sec",
            "range": "stddev: 0.00028265575795265933",
            "extra": "mean: 3.6647375999905307 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 299.1153006922125,
            "unit": "iter/sec",
            "range": "stddev: 0.0002135623588145169",
            "extra": "mean: 3.3431924000069557 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 416.58000559634553,
            "unit": "iter/sec",
            "range": "stddev: 0.00006459600809796914",
            "extra": "mean: 2.4004992716068383 msec\nrounds: 324"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 282.66731981039925,
            "unit": "iter/sec",
            "range": "stddev: 0.000058961232853177725",
            "extra": "mean: 3.5377276746061623 msec\nrounds: 252"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 306.4615218031101,
            "unit": "iter/sec",
            "range": "stddev: 0.0000785124092723873",
            "extra": "mean: 3.263052386206129 msec\nrounds: 290"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 400.7728733785353,
            "unit": "iter/sec",
            "range": "stddev: 0.00006569938224200916",
            "extra": "mean: 2.4951788567173976 msec\nrounds: 335"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 242.7775424468864,
            "unit": "iter/sec",
            "range": "stddev: 0.00010550170409652809",
            "extra": "mean: 4.1189971276637944 msec\nrounds: 47"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 247.72178940548346,
            "unit": "iter/sec",
            "range": "stddev: 0.00014784498903338178",
            "extra": "mean: 4.036786599999687 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 400.6360446700398,
            "unit": "iter/sec",
            "range": "stddev: 0.00004921137789728019",
            "extra": "mean: 2.496031031914742 msec\nrounds: 376"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 245.0808018240702,
            "unit": "iter/sec",
            "range": "stddev: 0.00008994992438531977",
            "extra": "mean: 4.08028696069733 msec\nrounds: 229"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 251.00857231498165,
            "unit": "iter/sec",
            "range": "stddev: 0.00007632474307558043",
            "extra": "mean: 3.9839276833347985 msec\nrounds: 240"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 383.4357347304276,
            "unit": "iter/sec",
            "range": "stddev: 0.00005240157045860269",
            "extra": "mean: 2.6079989667709103 msec\nrounds: 331"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 211.67324279439063,
            "unit": "iter/sec",
            "range": "stddev: 0.00010878272082125334",
            "extra": "mean: 4.7242626739145885 msec\nrounds: 46"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 217.35949425878823,
            "unit": "iter/sec",
            "range": "stddev: 0.00026918236195368715",
            "extra": "mean: 4.600673199990979 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 385.4832297438224,
            "unit": "iter/sec",
            "range": "stddev: 0.000056667804026533325",
            "extra": "mean: 2.5941465745852605 msec\nrounds: 362"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 210.36530848472023,
            "unit": "iter/sec",
            "range": "stddev: 0.00021017738641474298",
            "extra": "mean: 4.7536355076941526 msec\nrounds: 195"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 218.48641910429916,
            "unit": "iter/sec",
            "range": "stddev: 0.00007577449641780302",
            "extra": "mean: 4.576943519416777 msec\nrounds: 206"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_to_matrix",
            "value": 596.2090131914973,
            "unit": "iter/sec",
            "range": "stddev: 0.000022389148042205212",
            "extra": "mean: 1.677264143738814 msec\nrounds: 487"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_from_matrix",
            "value": 10177.451259182808,
            "unit": "iter/sec",
            "range": "stddev: 0.000009106827348992534",
            "extra": "mean: 98.2564273248403 usec\nrounds: 3323"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body",
            "value": 44880.785405402814,
            "unit": "iter/sec",
            "range": "stddev: 0.000009568336728576802",
            "extra": "mean: 22.281250004141384 usec\nrounds: 8"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body_vmap",
            "value": 34719.46668482264,
            "unit": "iter/sec",
            "range": "stddev: 0.00001172037814528683",
            "extra": "mean: 28.802285734335392 usec\nrounds: 7"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body",
            "value": 328.59665634540255,
            "unit": "iter/sec",
            "range": "stddev: 0.00012570415986262024",
            "extra": "mean: 3.043244599996342 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body_matrix",
            "value": 3.8059728817718654,
            "unit": "iter/sec",
            "range": "stddev: 0.001918473589039173",
            "extra": "mean: 262.7449093999985 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=1,]",
            "value": 528.9435918431428,
            "unit": "iter/sec",
            "range": "stddev: 0.00013481705196195567",
            "extra": "mean: 1.8905607619054925 msec\nrounds: 105"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.417030004414418,
            "unit": "iter/sec",
            "range": "stddev: 0.0024377130377992973",
            "extra": "mean: 226.39646979997678 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=1,]",
            "value": 1112.5863379876416,
            "unit": "iter/sec",
            "range": "stddev: 0.000025918270154332053",
            "extra": "mean: 898.8066506449478 usec\nrounds: 853"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=10000,]",
            "value": 15.91899921303466,
            "unit": "iter/sec",
            "range": "stddev: 0.0013962208041066455",
            "extra": "mean: 62.818019312494755 msec\nrounds: 16"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 545.0989353639197,
            "unit": "iter/sec",
            "range": "stddev: 0.00002990559051325306",
            "extra": "mean: 1.8345293581107045 msec\nrounds: 444"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.394106911009755,
            "unit": "iter/sec",
            "range": "stddev: 0.0019822763433635713",
            "extra": "mean: 227.5775306000014 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1118.8361391763963,
            "unit": "iter/sec",
            "range": "stddev: 0.000041263351255833286",
            "extra": "mean: 893.7859307406046 usec\nrounds: 823"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 16.011345495314597,
            "unit": "iter/sec",
            "range": "stddev: 0.00028542783743036425",
            "extra": "mean: 62.45571306250497 msec\nrounds: 16"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=1,]",
            "value": 544.1016588081742,
            "unit": "iter/sec",
            "range": "stddev: 0.000049540529628555134",
            "extra": "mean: 1.8378918421062105 msec\nrounds: 418"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.437411939338653,
            "unit": "iter/sec",
            "range": "stddev: 0.0010677963715369219",
            "extra": "mean: 225.35658479997664 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=1,]",
            "value": 1109.87364629666,
            "unit": "iter/sec",
            "range": "stddev: 0.0000577657561992635",
            "extra": "mean: 901.0034640760435 usec\nrounds: 849"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=10000,]",
            "value": 16.321062613469675,
            "unit": "iter/sec",
            "range": "stddev: 0.0003310313889469157",
            "extra": "mean: 61.270520411747334 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 541.2266465184621,
            "unit": "iter/sec",
            "range": "stddev: 0.000029610493864059415",
            "extra": "mean: 1.8476547790702478 msec\nrounds: 430"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.41759983294585,
            "unit": "iter/sec",
            "range": "stddev: 0.001096191237692042",
            "extra": "mean: 226.36726679997992 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1121.7674025705655,
            "unit": "iter/sec",
            "range": "stddev: 0.000021472920234742954",
            "extra": "mean: 891.4504002420363 usec\nrounds: 822"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 16.263861770704988,
            "unit": "iter/sec",
            "range": "stddev: 0.0002617074980163433",
            "extra": "mean: 61.48601200000565 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/orbits/tests/test_benchmarks.py::test_benchmark_iterate_real_orbits",
            "value": 16683.227252030058,
            "unit": "iter/sec",
            "range": "stddev: 0.0000031516586740677517",
            "extra": "mean: 59.94044107253394 usec\nrounds: 14136"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1342.2965000611007,
            "unit": "iter/sec",
            "range": "stddev: 0.000020886400509280058",
            "extra": "mean: 744.9918851419791 usec\nrounds: 949"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 15.985278932834946,
            "unit": "iter/sec",
            "range": "stddev: 0.00032108999997396663",
            "extra": "mean: 62.557557125007435 msec\nrounds: 16"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1367.283523321526,
            "unit": "iter/sec",
            "range": "stddev: 0.000021902644974786203",
            "extra": "mean: 731.3772037351198 usec\nrounds: 1178"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 17.60367384448226,
            "unit": "iter/sec",
            "range": "stddev: 0.00048215600893362464",
            "extra": "mean: 56.80632400000086 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1366.2204137496478,
            "unit": "iter/sec",
            "range": "stddev: 0.000019701914513221867",
            "extra": "mean: 731.9463169602766 usec\nrounds: 1079"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 16.55956517870474,
            "unit": "iter/sec",
            "range": "stddev: 0.0013003218767776068",
            "extra": "mean: 60.38805905881993 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1346.9529242602935,
            "unit": "iter/sec",
            "range": "stddev: 0.000024997612226567787",
            "extra": "mean: 742.4164438034613 usec\nrounds: 1041"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 15.82545973850611,
            "unit": "iter/sec",
            "range": "stddev: 0.000832710081483165",
            "extra": "mean: 63.18931750000445 msec\nrounds: 16"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1360.6529841455078,
            "unit": "iter/sec",
            "range": "stddev: 0.000020082875564771268",
            "extra": "mean: 734.9412463369574 usec\nrounds: 1092"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 17.565130714259126,
            "unit": "iter/sec",
            "range": "stddev: 0.00020578991625002404",
            "extra": "mean: 56.930973999995 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1348.925551857356,
            "unit": "iter/sec",
            "range": "stddev: 0.000025875230813499832",
            "extra": "mean: 741.3307566329994 usec\nrounds: 1093"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 16.494529113986506,
            "unit": "iter/sec",
            "range": "stddev: 0.0004699205042174771",
            "extra": "mean: 60.62616235294961 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1343.205029491297,
            "unit": "iter/sec",
            "range": "stddev: 0.000024675872582089165",
            "extra": "mean: 744.4879806463525 usec\nrounds: 1085"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 16.135129683770185,
            "unit": "iter/sec",
            "range": "stddev: 0.00036713319816820516",
            "extra": "mean: 61.976570352940406 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1349.3807050481566,
            "unit": "iter/sec",
            "range": "stddev: 0.000026691064523312392",
            "extra": "mean: 741.0807018796909 usec\nrounds: 1117"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 16.543100056790674,
            "unit": "iter/sec",
            "range": "stddev: 0.0002846882772369203",
            "extra": "mean: 60.4481624705834 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1363.865133998599,
            "unit": "iter/sec",
            "range": "stddev: 0.000025691010725070102",
            "extra": "mean: 733.2103263525668 usec\nrounds: 1146"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 17.489736330423447,
            "unit": "iter/sec",
            "range": "stddev: 0.0009403303603956867",
            "extra": "mean: 57.1763908333196 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1349.184619855981,
            "unit": "iter/sec",
            "range": "stddev: 0.000025313230834692882",
            "extra": "mean: 741.1884076374553 usec\nrounds: 1126"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 16.13565816554195,
            "unit": "iter/sec",
            "range": "stddev: 0.00047387573262466077",
            "extra": "mean: 61.97454047059088 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1358.1331940062366,
            "unit": "iter/sec",
            "range": "stddev: 0.00003242068320250065",
            "extra": "mean: 736.3048075205265 usec\nrounds: 1117"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 16.501253458992256,
            "unit": "iter/sec",
            "range": "stddev: 0.0014044888857195547",
            "extra": "mean: 60.60145688235921 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1350.1227688983515,
            "unit": "iter/sec",
            "range": "stddev: 0.000035448037746400024",
            "extra": "mean: 740.6733839589726 usec\nrounds: 1172"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 17.511518400664333,
            "unit": "iter/sec",
            "range": "stddev: 0.00101493912266977",
            "extra": "mean: 57.105270777779225 msec\nrounds: 18"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "moeyensj@users.noreply.github.com",
            "name": "Joachim Moeyens",
            "username": "moeyensj"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "dea97f3be3b97f70593623472b28b7ff74a95ab9",
          "message": "Add mean impact time calculations (#136)\n\n* Add mean impact time calculations\n\n* Point to quivr with table nulls support\n\n---------\n\nCo-authored-by: Alec Koumjian <akoumjian@gmail.com>",
          "timestamp": "2025-01-13T09:55:31-05:00",
          "tree_id": "26949c4af3c98ae207fd7a3d4c5dc7b933526324",
          "url": "https://github.com/B612-Asteroid-Institute/adam_core/commit/dea97f3be3b97f70593623472b28b7ff74a95ab9"
        },
        "date": 1736780386573,
        "tool": "pytest",
        "benches": [
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 422.32712593339517,
            "unit": "iter/sec",
            "range": "stddev: 0.00009268494879190562",
            "extra": "mean: 2.367832750003629 msec\nrounds: 8"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 281.03015537799536,
            "unit": "iter/sec",
            "range": "stddev: 0.0002210777245775051",
            "extra": "mean: 3.5583370000097148 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 298.6465339086535,
            "unit": "iter/sec",
            "range": "stddev: 0.00019682508915107343",
            "extra": "mean: 3.3484399999963443 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 431.6232212822085,
            "unit": "iter/sec",
            "range": "stddev: 0.00006046776364383417",
            "extra": "mean: 2.3168354960822866 msec\nrounds: 383"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 293.49566537133273,
            "unit": "iter/sec",
            "range": "stddev: 0.0001002874764673134",
            "extra": "mean: 3.4072053457239075 msec\nrounds: 269"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 317.7295125128115,
            "unit": "iter/sec",
            "range": "stddev: 0.00004394002171144035",
            "extra": "mean: 3.147331175160123 msec\nrounds: 314"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 415.2481639254212,
            "unit": "iter/sec",
            "range": "stddev: 0.00007041268331631598",
            "extra": "mean: 2.408198486771878 msec\nrounds: 378"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 254.60095625790515,
            "unit": "iter/sec",
            "range": "stddev: 0.00007960013320885758",
            "extra": "mean: 3.9277150199978905 msec\nrounds: 50"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 259.8693844503664,
            "unit": "iter/sec",
            "range": "stddev: 0.00019459679849813762",
            "extra": "mean: 3.8480869999943934 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 418.7870933793285,
            "unit": "iter/sec",
            "range": "stddev: 0.00004605779461593873",
            "extra": "mean: 2.3878481830246407 msec\nrounds: 377"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 248.9009993378514,
            "unit": "iter/sec",
            "range": "stddev: 0.000211935178044661",
            "extra": "mean: 4.017661651260096 msec\nrounds: 238"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 264.42609913595265,
            "unit": "iter/sec",
            "range": "stddev: 0.00005453277474253739",
            "extra": "mean: 3.7817749581740707 msec\nrounds: 263"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 401.38244315592647,
            "unit": "iter/sec",
            "range": "stddev: 0.000050545659057996137",
            "extra": "mean: 2.491389489130013 msec\nrounds: 368"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 223.28315237396188,
            "unit": "iter/sec",
            "range": "stddev: 0.00007584393884124564",
            "extra": "mean: 4.478618244896362 msec\nrounds: 49"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 227.87022371477866,
            "unit": "iter/sec",
            "range": "stddev: 0.00020980157170383842",
            "extra": "mean: 4.38846280000007 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 398.83990161105345,
            "unit": "iter/sec",
            "range": "stddev: 0.00009298641268518886",
            "extra": "mean: 2.507271704663077 msec\nrounds: 386"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 217.13774702605036,
            "unit": "iter/sec",
            "range": "stddev: 0.0003422301925476557",
            "extra": "mean: 4.605371538095715 msec\nrounds: 210"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 229.917152504066,
            "unit": "iter/sec",
            "range": "stddev: 0.00008978597094885174",
            "extra": "mean: 4.349392766519737 msec\nrounds: 227"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_to_matrix",
            "value": 595.612661528453,
            "unit": "iter/sec",
            "range": "stddev: 0.000021904002507126948",
            "extra": "mean: 1.6789434889342574 msec\nrounds: 497"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_from_matrix",
            "value": 10620.156714785217,
            "unit": "iter/sec",
            "range": "stddev: 0.000007316796871549138",
            "extra": "mean: 94.16056908160456 usec\nrounds: 3713"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body",
            "value": 45749.839875951795,
            "unit": "iter/sec",
            "range": "stddev: 0.000010360417682285552",
            "extra": "mean: 21.85799999981302 usec\nrounds: 9"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body_vmap",
            "value": 36022.68398842877,
            "unit": "iter/sec",
            "range": "stddev: 0.00001018403086479538",
            "extra": "mean: 27.760285722219386 usec\nrounds: 7"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body",
            "value": 325.076456357879,
            "unit": "iter/sec",
            "range": "stddev: 0.00012246301659925995",
            "extra": "mean: 3.076199399993129 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body_matrix",
            "value": 3.8752775586171158,
            "unit": "iter/sec",
            "range": "stddev: 0.0029989156350450465",
            "extra": "mean: 258.0460327999958 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=1,]",
            "value": 542.537915346541,
            "unit": "iter/sec",
            "range": "stddev: 0.000039632841626501485",
            "extra": "mean: 1.8431891517872248 msec\nrounds: 112"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.390571843031056,
            "unit": "iter/sec",
            "range": "stddev: 0.00403900100542595",
            "extra": "mean: 227.76076460000354 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=1,]",
            "value": 1116.5078100070139,
            "unit": "iter/sec",
            "range": "stddev: 0.00002182976040959447",
            "extra": "mean: 895.6498029276821 usec\nrounds: 888"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=10000,]",
            "value": 16.067718712395717,
            "unit": "iter/sec",
            "range": "stddev: 0.00044514227968102724",
            "extra": "mean: 62.23658864705746 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 549.3360494552849,
            "unit": "iter/sec",
            "range": "stddev: 0.00003178322662699609",
            "extra": "mean: 1.8203793488368152 msec\nrounds: 473"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.423368497237585,
            "unit": "iter/sec",
            "range": "stddev: 0.0007260701593201211",
            "extra": "mean: 226.07205359998943 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1119.4457778648955,
            "unit": "iter/sec",
            "range": "stddev: 0.000019557897352605555",
            "extra": "mean: 893.2991840902622 usec\nrounds: 880"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 15.888911980257364,
            "unit": "iter/sec",
            "range": "stddev: 0.0011989844153395168",
            "extra": "mean: 62.93697147057908 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=1,]",
            "value": 548.6529737806092,
            "unit": "iter/sec",
            "range": "stddev: 0.000027823533575734294",
            "extra": "mean: 1.8226457301585168 msec\nrounds: 441"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.422985584852196,
            "unit": "iter/sec",
            "range": "stddev: 0.0020637132669113108",
            "extra": "mean: 226.09162539999943 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=1,]",
            "value": 1123.180991012264,
            "unit": "iter/sec",
            "range": "stddev: 0.00001821548717199875",
            "extra": "mean: 890.3284581933251 usec\nrounds: 897"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=10000,]",
            "value": 16.35559756819125,
            "unit": "iter/sec",
            "range": "stddev: 0.000321107451759766",
            "extra": "mean: 61.141147294111924 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 549.3425814815205,
            "unit": "iter/sec",
            "range": "stddev: 0.00003472136453356551",
            "extra": "mean: 1.8203577033899372 msec\nrounds: 472"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.435256052609804,
            "unit": "iter/sec",
            "range": "stddev: 0.0006202296954068367",
            "extra": "mean: 225.46612600000344 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1114.5652202107292,
            "unit": "iter/sec",
            "range": "stddev: 0.000020701245758379146",
            "extra": "mean: 897.2108422788677 usec\nrounds: 913"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 16.26704769569149,
            "unit": "iter/sec",
            "range": "stddev: 0.0006085542600574195",
            "extra": "mean: 61.47396987499221 msec\nrounds: 16"
          },
          {
            "name": "src/adam_core/orbits/tests/test_benchmarks.py::test_benchmark_iterate_real_orbits",
            "value": 17282.584291706204,
            "unit": "iter/sec",
            "range": "stddev: 0.000004269225390221862",
            "extra": "mean: 57.861716923891606 usec\nrounds: 14512"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1361.3353164217463,
            "unit": "iter/sec",
            "range": "stddev: 0.00003558696901033782",
            "extra": "mean: 734.5728770399405 usec\nrounds: 919"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 15.909843283029142,
            "unit": "iter/sec",
            "range": "stddev: 0.0010587174577727841",
            "extra": "mean: 62.854170352934226 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1370.9879844917664,
            "unit": "iter/sec",
            "range": "stddev: 0.00003334731630316387",
            "extra": "mean: 729.4009949844353 usec\nrounds: 1196"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 17.425148295806096,
            "unit": "iter/sec",
            "range": "stddev: 0.000397873404004078",
            "extra": "mean: 57.38832077777386 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1371.0213333465913,
            "unit": "iter/sec",
            "range": "stddev: 0.000018160242097555323",
            "extra": "mean: 729.3832529644542 usec\nrounds: 1012"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 16.755087914308053,
            "unit": "iter/sec",
            "range": "stddev: 0.0005512034602037834",
            "extra": "mean: 59.68336335293396 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1363.285084303174,
            "unit": "iter/sec",
            "range": "stddev: 0.000055482013048908215",
            "extra": "mean: 733.5222922292422 usec\nrounds: 1184"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 15.960286322789157,
            "unit": "iter/sec",
            "range": "stddev: 0.0003348688264871213",
            "extra": "mean: 62.65551756249721 msec\nrounds: 16"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1373.372916445293,
            "unit": "iter/sec",
            "range": "stddev: 0.00003437940279985751",
            "extra": "mean: 728.1343530410549 usec\nrounds: 1184"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 17.468211702558015,
            "unit": "iter/sec",
            "range": "stddev: 0.00033827099884589103",
            "extra": "mean: 57.246844555562696 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1364.0325246268176,
            "unit": "iter/sec",
            "range": "stddev: 0.00003394875365737024",
            "extra": "mean: 733.1203486321469 usec\nrounds: 1133"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 16.585851956660644,
            "unit": "iter/sec",
            "range": "stddev: 0.0012185187312008668",
            "extra": "mean: 60.292350529416986 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1379.6920181425878,
            "unit": "iter/sec",
            "range": "stddev: 0.0000160134225744971",
            "extra": "mean: 724.7994384618179 usec\nrounds: 1170"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 16.339486424453685,
            "unit": "iter/sec",
            "range": "stddev: 0.0011335091004579605",
            "extra": "mean: 61.20143399999399 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1376.8079152441521,
            "unit": "iter/sec",
            "range": "stddev: 0.00002091545366224614",
            "extra": "mean: 726.3177302569967 usec\nrounds: 1127"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 16.730723769190067,
            "unit": "iter/sec",
            "range": "stddev: 0.0010712565369253432",
            "extra": "mean: 59.77027735294501 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1385.9482704895552,
            "unit": "iter/sec",
            "range": "stddev: 0.000017920505785303635",
            "extra": "mean: 721.527650990013 usec\nrounds: 1212"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 17.529568591445386,
            "unit": "iter/sec",
            "range": "stddev: 0.00029219315155349046",
            "extra": "mean: 57.0464694999973 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1375.9240025204242,
            "unit": "iter/sec",
            "range": "stddev: 0.000018607667168661988",
            "extra": "mean: 726.784326872847 usec\nrounds: 1135"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 16.06465730487366,
            "unit": "iter/sec",
            "range": "stddev: 0.001125807884066717",
            "extra": "mean: 62.24844894118111 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1376.9969750586165,
            "unit": "iter/sec",
            "range": "stddev: 0.00002596278824436238",
            "extra": "mean: 726.2180078191033 usec\nrounds: 1151"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 16.42540397230291,
            "unit": "iter/sec",
            "range": "stddev: 0.0011188161931948463",
            "extra": "mean: 60.88130323529546 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1367.2540862561166,
            "unit": "iter/sec",
            "range": "stddev: 0.0000771828391014479",
            "extra": "mean: 731.3929503317485 usec\nrounds: 1208"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 17.43730591515666,
            "unit": "iter/sec",
            "range": "stddev: 0.0009769609361371492",
            "extra": "mean: 57.34830855555452 msec\nrounds: 18"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "moeyensj@users.noreply.github.com",
            "name": "Joachim Moeyens",
            "username": "moeyensj"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "dea97f3be3b97f70593623472b28b7ff74a95ab9",
          "message": "Add mean impact time calculations (#136)\n\n* Add mean impact time calculations\n\n* Point to quivr with table nulls support\n\n---------\n\nCo-authored-by: Alec Koumjian <akoumjian@gmail.com>",
          "timestamp": "2025-01-13T09:55:31-05:00",
          "tree_id": "26949c4af3c98ae207fd7a3d4c5dc7b933526324",
          "url": "https://github.com/B612-Asteroid-Institute/adam_core/commit/dea97f3be3b97f70593623472b28b7ff74a95ab9"
        },
        "date": 1736780386463,
        "tool": "pytest",
        "benches": [
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 419.72583403704186,
            "unit": "iter/sec",
            "range": "stddev: 0.00010469493778436077",
            "extra": "mean: 2.3825076249934796 msec\nrounds: 8"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 282.0762277160188,
            "unit": "iter/sec",
            "range": "stddev: 0.0002148325048573322",
            "extra": "mean: 3.5451409999950556 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 303.1385510502325,
            "unit": "iter/sec",
            "range": "stddev: 0.00013954797309427008",
            "extra": "mean: 3.298821600009205 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 443.0150342374374,
            "unit": "iter/sec",
            "range": "stddev: 0.00008207203245974149",
            "extra": "mean: 2.257259737745248 msec\nrounds: 408"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 302.6343122915703,
            "unit": "iter/sec",
            "range": "stddev: 0.00008861411885430098",
            "extra": "mean: 3.3043179817513852 msec\nrounds: 274"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 325.3336130465521,
            "unit": "iter/sec",
            "range": "stddev: 0.00006201147412140384",
            "extra": "mean: 3.0737678490568685 msec\nrounds: 318"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 421.7911060766906,
            "unit": "iter/sec",
            "range": "stddev: 0.00010602474322424902",
            "extra": "mean: 2.3708418351954976 msec\nrounds: 358"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 249.5298969031905,
            "unit": "iter/sec",
            "range": "stddev: 0.00006500844157638818",
            "extra": "mean: 4.007535819998225 msec\nrounds: 50"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 256.34904814509895,
            "unit": "iter/sec",
            "range": "stddev: 0.0001557262938437763",
            "extra": "mean: 3.900931200001878 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 422.946443411736,
            "unit": "iter/sec",
            "range": "stddev: 0.00007661330135865054",
            "extra": "mean: 2.3643655492960503 msec\nrounds: 355"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 258.8257080501504,
            "unit": "iter/sec",
            "range": "stddev: 0.00007996392745630108",
            "extra": "mean: 3.863603841880493 msec\nrounds: 234"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 271.3763476573396,
            "unit": "iter/sec",
            "range": "stddev: 0.00008018494848243303",
            "extra": "mean: 3.684919517240597 msec\nrounds: 261"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 415.3529909016622,
            "unit": "iter/sec",
            "range": "stddev: 0.0000943782058359263",
            "extra": "mean: 2.407590704545467 msec\nrounds: 352"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 230.62518290610527,
            "unit": "iter/sec",
            "range": "stddev: 0.00012088850064056894",
            "extra": "mean: 4.336039921568891 msec\nrounds: 51"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 228.64730289188762,
            "unit": "iter/sec",
            "range": "stddev: 0.00019344876818149013",
            "extra": "mean: 4.373548200010191 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 404.34061388101196,
            "unit": "iter/sec",
            "range": "stddev: 0.00009575186399360681",
            "extra": "mean: 2.4731623924730863 msec\nrounds: 372"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 226.02385671407998,
            "unit": "iter/sec",
            "range": "stddev: 0.00011479946177558475",
            "extra": "mean: 4.424311727699609 msec\nrounds: 213"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 232.4661945094116,
            "unit": "iter/sec",
            "range": "stddev: 0.00008373147707351867",
            "extra": "mean: 4.301700735930076 msec\nrounds: 231"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_to_matrix",
            "value": 650.7713290418983,
            "unit": "iter/sec",
            "range": "stddev: 0.00010026668559990472",
            "extra": "mean: 1.5366380714286467 msec\nrounds: 560"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_from_matrix",
            "value": 10667.531296959649,
            "unit": "iter/sec",
            "range": "stddev: 0.000007701336902732484",
            "extra": "mean: 93.74240132625717 usec\nrounds: 2564"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body",
            "value": 36828.19238758412,
            "unit": "iter/sec",
            "range": "stddev: 0.00001311890422102004",
            "extra": "mean: 27.153111113244044 usec\nrounds: 9"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body_vmap",
            "value": 34547.48551972655,
            "unit": "iter/sec",
            "range": "stddev: 0.000011192167195628408",
            "extra": "mean: 28.94566666592861 usec\nrounds: 6"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body",
            "value": 339.0082753616611,
            "unit": "iter/sec",
            "range": "stddev: 0.000096890127887011",
            "extra": "mean: 2.949780499998648 msec\nrounds: 6"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body_matrix",
            "value": 3.8660946478595695,
            "unit": "iter/sec",
            "range": "stddev: 0.0007827157400803292",
            "extra": "mean: 258.65895459999706 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=1,]",
            "value": 544.7820548221736,
            "unit": "iter/sec",
            "range": "stddev: 0.00026992655322386004",
            "extra": "mean: 1.8355964392520558 msec\nrounds: 107"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.638366848250331,
            "unit": "iter/sec",
            "range": "stddev: 0.0016679466906247924",
            "extra": "mean: 215.59312420000083 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=1,]",
            "value": 1140.706637508688,
            "unit": "iter/sec",
            "range": "stddev: 0.00008254852054321104",
            "extra": "mean: 876.6495846679805 usec\nrounds: 874"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=10000,]",
            "value": 16.75196950195503,
            "unit": "iter/sec",
            "range": "stddev: 0.0013333693237857647",
            "extra": "mean: 59.694473529413685 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 562.5672260984928,
            "unit": "iter/sec",
            "range": "stddev: 0.000058318025792958935",
            "extra": "mean: 1.777565335498095 msec\nrounds: 462"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.6096590033240625,
            "unit": "iter/sec",
            "range": "stddev: 0.0017650352351250905",
            "extra": "mean: 216.93578619999698 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1155.0658574753124,
            "unit": "iter/sec",
            "range": "stddev: 0.00004939008394854697",
            "extra": "mean: 865.751501118518 usec\nrounds: 894"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 16.650644404519888,
            "unit": "iter/sec",
            "range": "stddev: 0.0011664794014230506",
            "extra": "mean: 60.05773564706875 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=1,]",
            "value": 569.5152350824328,
            "unit": "iter/sec",
            "range": "stddev: 0.00007144083414412004",
            "extra": "mean: 1.755879278374806 msec\nrounds: 467"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.652528829425488,
            "unit": "iter/sec",
            "range": "stddev: 0.000956856182723734",
            "extra": "mean: 214.9368733999836 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=1,]",
            "value": 1171.261367041137,
            "unit": "iter/sec",
            "range": "stddev: 0.000023776436048546657",
            "extra": "mean: 853.7804013174441 usec\nrounds: 912"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=10000,]",
            "value": 17.139268637052798,
            "unit": "iter/sec",
            "range": "stddev: 0.00027309770096667184",
            "extra": "mean: 58.34554677777407 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 576.0759386891068,
            "unit": "iter/sec",
            "range": "stddev: 0.000029603677897495067",
            "extra": "mean: 1.7358822558629272 msec\nrounds: 469"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.677209421613316,
            "unit": "iter/sec",
            "range": "stddev: 0.0005260368602884194",
            "extra": "mean: 213.80269939999152 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1166.9814048890996,
            "unit": "iter/sec",
            "range": "stddev: 0.00002066023866870223",
            "extra": "mean: 856.9116832628811 usec\nrounds: 944"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 16.974325499056977,
            "unit": "iter/sec",
            "range": "stddev: 0.00038648134952487653",
            "extra": "mean: 58.91250288888096 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/orbits/tests/test_benchmarks.py::test_benchmark_iterate_real_orbits",
            "value": 19587.34324465574,
            "unit": "iter/sec",
            "range": "stddev: 0.0000040536570084082755",
            "extra": "mean: 51.05337602499218 usec\nrounds: 15366"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1421.1030546396296,
            "unit": "iter/sec",
            "range": "stddev: 0.00001932582065728487",
            "extra": "mean: 703.6787351453446 usec\nrounds: 993"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 16.976802167514936,
            "unit": "iter/sec",
            "range": "stddev: 0.0004930217164904562",
            "extra": "mean: 58.90390841176775 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1447.5827913257149,
            "unit": "iter/sec",
            "range": "stddev: 0.000017005059387740725",
            "extra": "mean: 690.8067752616672 usec\nrounds: 1237"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 18.84359101516439,
            "unit": "iter/sec",
            "range": "stddev: 0.00036282695154506915",
            "extra": "mean: 53.06844110526754 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1451.4294813685194,
            "unit": "iter/sec",
            "range": "stddev: 0.000017756147292856925",
            "extra": "mean: 688.9759460150437 usec\nrounds: 1167"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 17.782057396511053,
            "unit": "iter/sec",
            "range": "stddev: 0.000768231353938535",
            "extra": "mean: 56.23646227776805 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1438.822378784747,
            "unit": "iter/sec",
            "range": "stddev: 0.00001911770643740431",
            "extra": "mean: 695.0128207239983 usec\nrounds: 1216"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 16.903488988706894,
            "unit": "iter/sec",
            "range": "stddev: 0.000519820779652484",
            "extra": "mean: 59.15938423529563 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1446.5207403978527,
            "unit": "iter/sec",
            "range": "stddev: 0.000019385248307338112",
            "extra": "mean: 691.3139729506807 usec\nrounds: 1183"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 18.614894070638893,
            "unit": "iter/sec",
            "range": "stddev: 0.0013367247391780324",
            "extra": "mean: 53.720423882362624 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1436.0586944906518,
            "unit": "iter/sec",
            "range": "stddev: 0.00001853804610037837",
            "extra": "mean: 696.3503677366647 usec\nrounds: 1153"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 17.50956169440571,
            "unit": "iter/sec",
            "range": "stddev: 0.0013086264349524868",
            "extra": "mean: 57.111652333336195 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1424.338520116774,
            "unit": "iter/sec",
            "range": "stddev: 0.000023324277465461887",
            "extra": "mean: 702.0802891141464 usec\nrounds: 1176"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 17.262952080774436,
            "unit": "iter/sec",
            "range": "stddev: 0.0011246144481177618",
            "extra": "mean: 57.927519888889066 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1436.7501885630993,
            "unit": "iter/sec",
            "range": "stddev: 0.00001814329748252707",
            "extra": "mean: 696.0152209898818 usec\nrounds: 1172"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 17.76136699390256,
            "unit": "iter/sec",
            "range": "stddev: 0.00032577111909697613",
            "extra": "mean: 56.30197272221772 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1445.7850399770343,
            "unit": "iter/sec",
            "range": "stddev: 0.000032821263429064874",
            "extra": "mean: 691.6657541399685 usec\nrounds: 1208"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 18.7006742220018,
            "unit": "iter/sec",
            "range": "stddev: 0.0004312807862096386",
            "extra": "mean: 53.47400784210633 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1415.996347908803,
            "unit": "iter/sec",
            "range": "stddev: 0.00004033688801456775",
            "extra": "mean: 706.2165107112301 usec\nrounds: 1167"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 16.87133033209791,
            "unit": "iter/sec",
            "range": "stddev: 0.002252779214746943",
            "extra": "mean: 59.27214868749786 msec\nrounds: 16"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1446.6414397547317,
            "unit": "iter/sec",
            "range": "stddev: 0.000016586142353251386",
            "extra": "mean: 691.2562937292487 usec\nrounds: 1212"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 17.419194952977445,
            "unit": "iter/sec",
            "range": "stddev: 0.002165812667355182",
            "extra": "mean: 57.40793433332985 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1398.6839334625427,
            "unit": "iter/sec",
            "range": "stddev: 0.00007743507491803904",
            "extra": "mean: 714.9578086054282 usec\nrounds: 674"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 18.42810967362445,
            "unit": "iter/sec",
            "range": "stddev: 0.0007448150784806733",
            "extra": "mean: 54.2649255789522 msec\nrounds: 19"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "akoumjian@users.noreply.github.com",
            "name": "Alec Koumjian",
            "username": "akoumjian"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ee57577ff786c573c856d591b1e993d1649576f3",
          "message": "Initial parsing ADES attempt (#137)\n\n* Initial parsing ADES attempt\n\n* linting\n\n* ensure null handling in ades\n\n* more linting\n\n* Fix buggy whitelist logic and change back test strings",
          "timestamp": "2025-01-22T16:33:23-05:00",
          "tree_id": "b1a3bcb41c2ded3126e32e283fa841d018d5d31e",
          "url": "https://github.com/B612-Asteroid-Institute/adam_core/commit/ee57577ff786c573c856d591b1e993d1649576f3"
        },
        "date": 1737581869579,
        "tool": "pytest",
        "benches": [
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 434.63027550706,
            "unit": "iter/sec",
            "range": "stddev: 0.00007811935370946555",
            "extra": "mean: 2.3008061250067158 msec\nrounds: 8"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 297.1596706067737,
            "unit": "iter/sec",
            "range": "stddev: 0.00023260280909540682",
            "extra": "mean: 3.3651942000005874 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 306.19667355479703,
            "unit": "iter/sec",
            "range": "stddev: 0.00019142054596500138",
            "extra": "mean: 3.265874799978974 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 439.87135299543917,
            "unit": "iter/sec",
            "range": "stddev: 0.00008924766701645742",
            "extra": "mean: 2.273391966060514 msec\nrounds: 383"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 302.9419572328257,
            "unit": "iter/sec",
            "range": "stddev: 0.00007898481717814183",
            "extra": "mean: 3.300962366303889 msec\nrounds: 273"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 324.11075558964,
            "unit": "iter/sec",
            "range": "stddev: 0.00010665218029592432",
            "extra": "mean: 3.0853650573266087 msec\nrounds: 314"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 422.2036778516531,
            "unit": "iter/sec",
            "range": "stddev: 0.00007801004328275324",
            "extra": "mean: 2.368525080331876 msec\nrounds: 361"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 260.9372500711031,
            "unit": "iter/sec",
            "range": "stddev: 0.00007145749104304457",
            "extra": "mean: 3.8323389999990756 msec\nrounds: 51"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 266.11150838531137,
            "unit": "iter/sec",
            "range": "stddev: 0.00018891831553284976",
            "extra": "mean: 3.7578232000100797 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 423.49911116997515,
            "unit": "iter/sec",
            "range": "stddev: 0.00008788709564355454",
            "extra": "mean: 2.361280044336719 msec\nrounds: 406"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 260.58138265674984,
            "unit": "iter/sec",
            "range": "stddev: 0.00009118645771329084",
            "extra": "mean: 3.8375726991872154 msec\nrounds: 246"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 268.94367721139764,
            "unit": "iter/sec",
            "range": "stddev: 0.00005473976781062535",
            "extra": "mean: 3.718250640315186 msec\nrounds: 253"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 404.5416751503441,
            "unit": "iter/sec",
            "range": "stddev: 0.00009083643211764479",
            "extra": "mean: 2.47193320596292 msec\nrounds: 369"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 229.40215591340367,
            "unit": "iter/sec",
            "range": "stddev: 0.0000772449510430248",
            "extra": "mean: 4.359156939996183 msec\nrounds: 50"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 233.1265671635505,
            "unit": "iter/sec",
            "range": "stddev: 0.00016965624359755997",
            "extra": "mean: 4.289515400012078 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 405.0753043119614,
            "unit": "iter/sec",
            "range": "stddev: 0.00012942456168069008",
            "extra": "mean: 2.4686767851684883 msec\nrounds: 391"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 227.42410189721792,
            "unit": "iter/sec",
            "range": "stddev: 0.00006751456172534688",
            "extra": "mean: 4.397071337900413 msec\nrounds: 219"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 232.06993393381228,
            "unit": "iter/sec",
            "range": "stddev: 0.00016785136110776647",
            "extra": "mean: 4.309045911501857 msec\nrounds: 226"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_to_matrix",
            "value": 680.7482377110306,
            "unit": "iter/sec",
            "range": "stddev: 0.00001900521026753309",
            "extra": "mean: 1.468971852740202 msec\nrounds: 584"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_from_matrix",
            "value": 10906.022108909901,
            "unit": "iter/sec",
            "range": "stddev: 0.000007545713623476469",
            "extra": "mean: 91.69246036857281 usec\nrounds: 3419"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body",
            "value": 45454.54545186207,
            "unit": "iter/sec",
            "range": "stddev: 0.000008685796536800191",
            "extra": "mean: 22.00000000129876 usec\nrounds: 9"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body_vmap",
            "value": 29036.00465663136,
            "unit": "iter/sec",
            "range": "stddev: 0.000013285673237768948",
            "extra": "mean: 34.43999998710622 usec\nrounds: 7"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body",
            "value": 339.5446807676164,
            "unit": "iter/sec",
            "range": "stddev: 0.00009623467621324301",
            "extra": "mean: 2.945120500015719 msec\nrounds: 6"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body_matrix",
            "value": 3.8543896123506873,
            "unit": "iter/sec",
            "range": "stddev: 0.004409215384728454",
            "extra": "mean: 259.44445180001594 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=1,]",
            "value": 557.6883611842619,
            "unit": "iter/sec",
            "range": "stddev: 0.00005300415191052697",
            "extra": "mean: 1.7931161372571607 msec\nrounds: 102"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.620709811409595,
            "unit": "iter/sec",
            "range": "stddev: 0.007454970427344932",
            "extra": "mean: 216.41696640000418 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=1,]",
            "value": 1160.6607971461801,
            "unit": "iter/sec",
            "range": "stddev: 0.000024376215745468584",
            "extra": "mean: 861.5781651786543 usec\nrounds: 896"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=10000,]",
            "value": 17.06817401451469,
            "unit": "iter/sec",
            "range": "stddev: 0.00023550004122539092",
            "extra": "mean: 58.58857538888489 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 564.7167201796219,
            "unit": "iter/sec",
            "range": "stddev: 0.000041418501727671754",
            "extra": "mean: 1.7707993481792528 msec\nrounds: 494"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.650435315530586,
            "unit": "iter/sec",
            "range": "stddev: 0.0011968491836729747",
            "extra": "mean: 215.0336328000094 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1155.6198263018266,
            "unit": "iter/sec",
            "range": "stddev: 0.00003621131986305582",
            "extra": "mean: 865.3364863081005 usec\nrounds: 913"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 16.86878795472271,
            "unit": "iter/sec",
            "range": "stddev: 0.0005519580157886332",
            "extra": "mean: 59.28108188235496 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=1,]",
            "value": 562.6180278073526,
            "unit": "iter/sec",
            "range": "stddev: 0.00006894393843602155",
            "extra": "mean: 1.7774048298758252 msec\nrounds: 482"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.695003290164815,
            "unit": "iter/sec",
            "range": "stddev: 0.0006924430749437043",
            "extra": "mean: 212.99239599998145 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=1,]",
            "value": 1160.8435249797978,
            "unit": "iter/sec",
            "range": "stddev: 0.00001944275304315152",
            "extra": "mean: 861.4425445646544 usec\nrounds: 920"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=10000,]",
            "value": 17.255162598702068,
            "unit": "iter/sec",
            "range": "stddev: 0.0013903694378845837",
            "extra": "mean: 57.95367005554731 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 566.1590143736278,
            "unit": "iter/sec",
            "range": "stddev: 0.00002960759777229917",
            "extra": "mean: 1.766288224000732 msec\nrounds: 500"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.643574300452273,
            "unit": "iter/sec",
            "range": "stddev: 0.002601779573161389",
            "extra": "mean: 215.35135120000177 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1153.1227126633614,
            "unit": "iter/sec",
            "range": "stddev: 0.00003693009754887994",
            "extra": "mean: 867.210392283667 usec\nrounds: 933"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 17.25165817547255,
            "unit": "iter/sec",
            "range": "stddev: 0.00038811711239715105",
            "extra": "mean: 57.965442499999476 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/orbits/tests/test_benchmarks.py::test_benchmark_iterate_real_orbits",
            "value": 19823.64444966004,
            "unit": "iter/sec",
            "range": "stddev: 0.000004651968362119316",
            "extra": "mean: 50.444811121355094 usec\nrounds: 16275"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1422.146768141597,
            "unit": "iter/sec",
            "range": "stddev: 0.00001993213798067526",
            "extra": "mean: 703.162305327149 usec\nrounds: 976"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 17.10521881936756,
            "unit": "iter/sec",
            "range": "stddev: 0.00023624528523418938",
            "extra": "mean: 58.46168999999811 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1411.2113308010794,
            "unit": "iter/sec",
            "range": "stddev: 0.00003058326151720034",
            "extra": "mean: 708.6110904681769 usec\nrounds: 1238"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 18.948652259953263,
            "unit": "iter/sec",
            "range": "stddev: 0.00043487400600566956",
            "extra": "mean: 52.774201894740266 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1424.199732091015,
            "unit": "iter/sec",
            "range": "stddev: 0.000017383200368719147",
            "extra": "mean: 702.1487067209291 usec\nrounds: 1190"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 17.89248614418265,
            "unit": "iter/sec",
            "range": "stddev: 0.0003153449361840053",
            "extra": "mean: 55.88938238888216 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1419.3588181502357,
            "unit": "iter/sec",
            "range": "stddev: 0.000023184803829676866",
            "extra": "mean: 704.5434792191868 usec\nrounds: 1179"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 16.94552812536717,
            "unit": "iter/sec",
            "range": "stddev: 0.0002374151045395029",
            "extra": "mean: 59.01261929411435 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1414.8178620592107,
            "unit": "iter/sec",
            "range": "stddev: 0.00007345229158117435",
            "extra": "mean: 706.8047603983032 usec\nrounds: 1202"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 18.909530358616596,
            "unit": "iter/sec",
            "range": "stddev: 0.0016906567448233332",
            "extra": "mean: 52.88338636841529 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1430.6422762425937,
            "unit": "iter/sec",
            "range": "stddev: 0.00001729317442781414",
            "extra": "mean: 698.9867534366293 usec\nrounds: 1164"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 17.681741576052808,
            "unit": "iter/sec",
            "range": "stddev: 0.002500673325791986",
            "extra": "mean: 56.55551494737067 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1422.366129515697,
            "unit": "iter/sec",
            "range": "stddev: 0.00003899736641776705",
            "extra": "mean: 703.0538616245672 usec\nrounds: 1243"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 17.454283974562976,
            "unit": "iter/sec",
            "range": "stddev: 0.0004579473152289643",
            "extra": "mean: 57.29252494444066 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1427.416602278363,
            "unit": "iter/sec",
            "range": "stddev: 0.000030823222342485765",
            "extra": "mean: 700.5663226866325 usec\nrounds: 1221"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 17.8930373195664,
            "unit": "iter/sec",
            "range": "stddev: 0.0002378612199658175",
            "extra": "mean: 55.88766077777526 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1442.5827528529605,
            "unit": "iter/sec",
            "range": "stddev: 0.00003252670828685024",
            "extra": "mean: 693.2011338845723 usec\nrounds: 1210"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 18.91070725606164,
            "unit": "iter/sec",
            "range": "stddev: 0.000412429691389634",
            "extra": "mean: 52.88009520000685 msec\nrounds: 20"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1421.0622656839546,
            "unit": "iter/sec",
            "range": "stddev: 0.000032411606536308075",
            "extra": "mean: 703.6989329378202 usec\nrounds: 1178"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 17.185106055056696,
            "unit": "iter/sec",
            "range": "stddev: 0.0021569466446088505",
            "extra": "mean: 58.18992311110883 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1423.4155614907027,
            "unit": "iter/sec",
            "range": "stddev: 0.000036288695291648335",
            "extra": "mean: 702.5355258535523 usec\nrounds: 1025"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 17.81831438140146,
            "unit": "iter/sec",
            "range": "stddev: 0.0004017780326732314",
            "extra": "mean: 56.12203144444391 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1432.6720420779084,
            "unit": "iter/sec",
            "range": "stddev: 0.00003086117375471963",
            "extra": "mean: 697.9964504295256 usec\nrounds: 1281"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 18.851038004330398,
            "unit": "iter/sec",
            "range": "stddev: 0.0014083972857667944",
            "extra": "mean: 53.04747673683981 msec\nrounds: 19"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "akoumjian@users.noreply.github.com",
            "name": "Alec Koumjian",
            "username": "akoumjian"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ee57577ff786c573c856d591b1e993d1649576f3",
          "message": "Initial parsing ADES attempt (#137)\n\n* Initial parsing ADES attempt\n\n* linting\n\n* ensure null handling in ades\n\n* more linting\n\n* Fix buggy whitelist logic and change back test strings",
          "timestamp": "2025-01-22T16:33:23-05:00",
          "tree_id": "b1a3bcb41c2ded3126e32e283fa841d018d5d31e",
          "url": "https://github.com/B612-Asteroid-Institute/adam_core/commit/ee57577ff786c573c856d591b1e993d1649576f3"
        },
        "date": 1737581890843,
        "tool": "pytest",
        "benches": [
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 425.1790866119504,
            "unit": "iter/sec",
            "range": "stddev: 0.0000810468498569029",
            "extra": "mean: 2.3519501111132333 msec\nrounds: 9"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 276.7001383451964,
            "unit": "iter/sec",
            "range": "stddev: 0.00022187324683025354",
            "extra": "mean: 3.6140205999913633 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 302.47532497059444,
            "unit": "iter/sec",
            "range": "stddev: 0.0002382006081517706",
            "extra": "mean: 3.306054799998037 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 436.3577238815049,
            "unit": "iter/sec",
            "range": "stddev: 0.000032558349497556705",
            "extra": "mean: 2.2916977178833093 msec\nrounds: 397"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 296.4754800168008,
            "unit": "iter/sec",
            "range": "stddev: 0.00004017919439343785",
            "extra": "mean: 3.3729602189811163 msec\nrounds: 274"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 318.4963003418054,
            "unit": "iter/sec",
            "range": "stddev: 0.00010780803711375453",
            "extra": "mean: 3.1397538964402885 msec\nrounds: 309"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 412.1495799068458,
            "unit": "iter/sec",
            "range": "stddev: 0.00005442618194427913",
            "extra": "mean: 2.426303577031476 msec\nrounds: 357"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 252.08757839386126,
            "unit": "iter/sec",
            "range": "stddev: 0.00011002160226093066",
            "extra": "mean: 3.9668753469383624 msec\nrounds: 49"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 259.07977454888527,
            "unit": "iter/sec",
            "range": "stddev: 0.00019330347052330112",
            "extra": "mean: 3.8598149999984344 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 412.19010338800183,
            "unit": "iter/sec",
            "range": "stddev: 0.00010302827574054895",
            "extra": "mean: 2.4260650408160873 msec\nrounds: 392"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 250.75887960027447,
            "unit": "iter/sec",
            "range": "stddev: 0.00013554882036529093",
            "extra": "mean: 3.9878946723404702 msec\nrounds: 235"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 260.5090045235101,
            "unit": "iter/sec",
            "range": "stddev: 0.00013704050062651288",
            "extra": "mean: 3.838638905511434 msec\nrounds: 254"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 386.6111729015356,
            "unit": "iter/sec",
            "range": "stddev: 0.00006075022081169118",
            "extra": "mean: 2.5865781179963103 msec\nrounds: 339"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 224.44537649234772,
            "unit": "iter/sec",
            "range": "stddev: 0.00007675046532835027",
            "extra": "mean: 4.45542704255302 msec\nrounds: 47"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 219.95911663895487,
            "unit": "iter/sec",
            "range": "stddev: 0.00027514528451094103",
            "extra": "mean: 4.546299399999043 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 390.75483349629263,
            "unit": "iter/sec",
            "range": "stddev: 0.00008459146487756928",
            "extra": "mean: 2.5591494059138435 msec\nrounds: 372"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 222.46109131271302,
            "unit": "iter/sec",
            "range": "stddev: 0.000045614424879592095",
            "extra": "mean: 4.495168094785178 msec\nrounds: 211"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 231.28514857769036,
            "unit": "iter/sec",
            "range": "stddev: 0.00004586964539713378",
            "extra": "mean: 4.323667153509828 msec\nrounds: 228"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_to_matrix",
            "value": 635.0206558767237,
            "unit": "iter/sec",
            "range": "stddev: 0.000014056237227365973",
            "extra": "mean: 1.5747519245958665 msec\nrounds: 557"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_from_matrix",
            "value": 10483.15578848713,
            "unit": "iter/sec",
            "range": "stddev: 0.000007049877006991464",
            "extra": "mean: 95.39112269019463 usec\nrounds: 3464"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body",
            "value": 43844.05159096214,
            "unit": "iter/sec",
            "range": "stddev: 0.000009093339422732506",
            "extra": "mean: 22.808111105455787 usec\nrounds: 9"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body_vmap",
            "value": 33952.069381768066,
            "unit": "iter/sec",
            "range": "stddev: 0.000011766152461200474",
            "extra": "mean: 29.453285711562266 usec\nrounds: 7"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body",
            "value": 333.3560904421004,
            "unit": "iter/sec",
            "range": "stddev: 0.0001111307219041416",
            "extra": "mean: 2.9997952000030637 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body_matrix",
            "value": 3.859761649620989,
            "unit": "iter/sec",
            "range": "stddev: 0.0021646476105737222",
            "extra": "mean: 259.0833556000007 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=1,]",
            "value": 530.1452840781238,
            "unit": "iter/sec",
            "range": "stddev: 0.000042931103522302695",
            "extra": "mean: 1.886275385319917 msec\nrounds: 109"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.504007424391182,
            "unit": "iter/sec",
            "range": "stddev: 0.0008432402949903343",
            "extra": "mean: 222.024500800012 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=1,]",
            "value": 1101.5264272887973,
            "unit": "iter/sec",
            "range": "stddev: 0.000018937697121143792",
            "extra": "mean: 907.8311470577372 usec\nrounds: 850"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=10000,]",
            "value": 16.67223192306474,
            "unit": "iter/sec",
            "range": "stddev: 0.0006163311652616585",
            "extra": "mean: 59.97997176470282 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 540.7129272915913,
            "unit": "iter/sec",
            "range": "stddev: 0.00003253014236206376",
            "extra": "mean: 1.8494101944426569 msec\nrounds: 468"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.491079454210583,
            "unit": "iter/sec",
            "range": "stddev: 0.0005070197307014915",
            "extra": "mean: 222.6636180000014 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1088.5656199789178,
            "unit": "iter/sec",
            "range": "stddev: 0.00002084390777927044",
            "extra": "mean: 918.6400724463143 usec\nrounds: 842"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 16.543864560634812,
            "unit": "iter/sec",
            "range": "stddev: 0.00045759601977049725",
            "extra": "mean: 60.44536911765122 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=1,]",
            "value": 545.364015244379,
            "unit": "iter/sec",
            "range": "stddev: 0.000026455866228108387",
            "extra": "mean: 1.8336376659393225 msec\nrounds: 458"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.48885707262805,
            "unit": "iter/sec",
            "range": "stddev: 0.0012504598532008073",
            "extra": "mean: 222.77385620000132 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=1,]",
            "value": 1089.4607555791085,
            "unit": "iter/sec",
            "range": "stddev: 0.00002150506199454241",
            "extra": "mean: 917.8852885512565 usec\nrounds: 856"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=10000,]",
            "value": 16.88056307831021,
            "unit": "iter/sec",
            "range": "stddev: 0.0012276014401500107",
            "extra": "mean: 59.23973005882116 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 544.1210123632775,
            "unit": "iter/sec",
            "range": "stddev: 0.00002903326481866715",
            "extra": "mean: 1.8378264710945567 msec\nrounds: 467"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.531730422337175,
            "unit": "iter/sec",
            "range": "stddev: 0.0005103252259110753",
            "extra": "mean: 220.66625920000433 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1106.1683704615532,
            "unit": "iter/sec",
            "range": "stddev: 0.00002046638367067744",
            "extra": "mean: 904.0215094766686 usec\nrounds: 897"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 16.83678282979595,
            "unit": "iter/sec",
            "range": "stddev: 0.00032341970814354736",
            "extra": "mean: 59.39376958823191 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/orbits/tests/test_benchmarks.py::test_benchmark_iterate_real_orbits",
            "value": 16825.048176577606,
            "unit": "iter/sec",
            "range": "stddev: 0.0000070479176063319645",
            "extra": "mean: 59.4351938553207 usec\nrounds: 14191"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1327.9381534880456,
            "unit": "iter/sec",
            "range": "stddev: 0.000027213222243250484",
            "extra": "mean: 753.0471184771197 usec\nrounds: 920"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 16.72930459453494,
            "unit": "iter/sec",
            "range": "stddev: 0.0004430119563169231",
            "extra": "mean: 59.7753477647048 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1353.9248790689628,
            "unit": "iter/sec",
            "range": "stddev: 0.00001842275351480894",
            "extra": "mean: 738.5934149372142 usec\nrounds: 1205"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 18.570976023267626,
            "unit": "iter/sec",
            "range": "stddev: 0.0009602382628394512",
            "extra": "mean: 53.84746600001515 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1339.866465289014,
            "unit": "iter/sec",
            "range": "stddev: 0.000027825928857295826",
            "extra": "mean: 746.3430318664601 usec\nrounds: 1067"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 17.555961284656654,
            "unit": "iter/sec",
            "range": "stddev: 0.0002670889267603494",
            "extra": "mean: 56.96070888889279 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1339.4426393916053,
            "unit": "iter/sec",
            "range": "stddev: 0.000019479761717032885",
            "extra": "mean: 746.5791894262936 usec\nrounds: 1135"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 16.652032305608195,
            "unit": "iter/sec",
            "range": "stddev: 0.00040933793418489634",
            "extra": "mean: 60.05273000000202 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1349.0007420564198,
            "unit": "iter/sec",
            "range": "stddev: 0.000018489213476804175",
            "extra": "mean: 741.2894365614637 usec\nrounds: 1198"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 18.54407784269657,
            "unit": "iter/sec",
            "range": "stddev: 0.00034672381398889507",
            "extra": "mean: 53.92557173684652 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1345.8771659793374,
            "unit": "iter/sec",
            "range": "stddev: 0.000020513796665464563",
            "extra": "mean: 743.0098565290263 usec\nrounds: 1164"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 17.40449069624612,
            "unit": "iter/sec",
            "range": "stddev: 0.00019831305936868126",
            "extra": "mean: 57.45643566666874 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1341.8489649133464,
            "unit": "iter/sec",
            "range": "stddev: 0.000016594218186524407",
            "extra": "mean: 745.2403557687863 usec\nrounds: 1144"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 16.983591829246528,
            "unit": "iter/sec",
            "range": "stddev: 0.00038722561657313293",
            "extra": "mean: 58.880359941172976 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1335.8571864223322,
            "unit": "iter/sec",
            "range": "stddev: 0.000018420117724555263",
            "extra": "mean: 748.583014834229 usec\nrounds: 1146"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 17.451263940027992,
            "unit": "iter/sec",
            "range": "stddev: 0.0004416566742669988",
            "extra": "mean: 57.30243972221968 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1341.8635599853499,
            "unit": "iter/sec",
            "range": "stddev: 0.00002016131549847761",
            "extra": "mean: 745.232249999335 usec\nrounds: 1188"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 18.48817572811809,
            "unit": "iter/sec",
            "range": "stddev: 0.0004350869847792747",
            "extra": "mean: 54.08862478947185 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1337.8181485313867,
            "unit": "iter/sec",
            "range": "stddev: 0.00001918664999759611",
            "extra": "mean: 747.4857484163805 usec\nrounds: 1105"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 16.862565203446984,
            "unit": "iter/sec",
            "range": "stddev: 0.0003151652965255032",
            "extra": "mean: 59.30295823529765 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1340.5255523214205,
            "unit": "iter/sec",
            "range": "stddev: 0.000020414662329391688",
            "extra": "mean: 745.9760824911363 usec\nrounds: 1188"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 17.340372189877087,
            "unit": "iter/sec",
            "range": "stddev: 0.0003144971814325723",
            "extra": "mean: 57.668889055551944 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1344.9309770753944,
            "unit": "iter/sec",
            "range": "stddev: 0.000028372233668791726",
            "extra": "mean: 743.5325805154251 usec\nrounds: 1242"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 18.520349126871036,
            "unit": "iter/sec",
            "range": "stddev: 0.0002789304012839218",
            "extra": "mean: 53.994662473673756 msec\nrounds: 19"
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
          "id": "b49e49a1bb4fff3d070c256d44396f7ac5165e57",
          "message": "Use new serialization assist for unit tests",
          "timestamp": "2025-01-24T11:11:59-05:00",
          "tree_id": "ba03f8e95a220334b6301081721ef4b517e1ad45",
          "url": "https://github.com/B612-Asteroid-Institute/adam_core/commit/b49e49a1bb4fff3d070c256d44396f7ac5165e57"
        },
        "date": 1737739798649,
        "tool": "pytest",
        "benches": [
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 422.38871377564016,
            "unit": "iter/sec",
            "range": "stddev: 0.00008909808281452865",
            "extra": "mean: 2.3674874999883855 msec\nrounds: 8"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 290.853534712152,
            "unit": "iter/sec",
            "range": "stddev: 0.00021642069857993445",
            "extra": "mean: 3.4381565999865416 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 314.9602331218964,
            "unit": "iter/sec",
            "range": "stddev: 0.0001756904250918925",
            "extra": "mean: 3.175003999990622 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 429.19822945266924,
            "unit": "iter/sec",
            "range": "stddev: 0.00004631379863178468",
            "extra": "mean: 2.329925734491589 msec\nrounds: 403"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 302.80866484528127,
            "unit": "iter/sec",
            "range": "stddev: 0.00011662230294539748",
            "extra": "mean: 3.302415406477703 msec\nrounds: 278"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 322.5486166207132,
            "unit": "iter/sec",
            "range": "stddev: 0.00016662361246765406",
            "extra": "mean: 3.1003078248384055 msec\nrounds: 314"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 412.8858057801072,
            "unit": "iter/sec",
            "range": "stddev: 0.00010579276427857503",
            "extra": "mean: 2.4219771811011963 msec\nrounds: 381"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 259.87908865572945,
            "unit": "iter/sec",
            "range": "stddev: 0.00005601407625286372",
            "extra": "mean: 3.84794330768465 msec\nrounds: 52"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 243.61459352684403,
            "unit": "iter/sec",
            "range": "stddev: 0.000303667528680227",
            "extra": "mean: 4.104844400012553 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 399.23188244474414,
            "unit": "iter/sec",
            "range": "stddev: 0.00011900184779099833",
            "extra": "mean: 2.5048099712787977 msec\nrounds: 383"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 258.3117737288828,
            "unit": "iter/sec",
            "range": "stddev: 0.00009495999266889093",
            "extra": "mean: 3.8712908264474755 msec\nrounds: 242"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 270.4834443874232,
            "unit": "iter/sec",
            "range": "stddev: 0.00014530370613500943",
            "extra": "mean: 3.697083946356672 msec\nrounds: 261"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 397.98914557438184,
            "unit": "iter/sec",
            "range": "stddev: 0.00007197172170993401",
            "extra": "mean: 2.5126313396230695 msec\nrounds: 371"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 226.9125839965179,
            "unit": "iter/sec",
            "range": "stddev: 0.00009407632517189211",
            "extra": "mean: 4.4069834399988395 msec\nrounds: 50"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 229.1006484520059,
            "unit": "iter/sec",
            "range": "stddev: 0.00017699373627951228",
            "extra": "mean: 4.364893799981928 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 393.9907578175649,
            "unit": "iter/sec",
            "range": "stddev: 0.00015509916249369974",
            "extra": "mean: 2.538130603721025 msec\nrounds: 376"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 223.5473925657572,
            "unit": "iter/sec",
            "range": "stddev: 0.00006325705024066344",
            "extra": "mean: 4.473324374409094 msec\nrounds: 211"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 232.2665304974993,
            "unit": "iter/sec",
            "range": "stddev: 0.0000540813043393017",
            "extra": "mean: 4.305398620533347 msec\nrounds: 224"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_to_matrix",
            "value": 677.0718219399021,
            "unit": "iter/sec",
            "range": "stddev: 0.00003986541583526508",
            "extra": "mean: 1.4769481871729135 msec\nrounds: 577"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_from_matrix",
            "value": 10550.746803299804,
            "unit": "iter/sec",
            "range": "stddev: 0.000007331162021590202",
            "extra": "mean: 94.78002066045642 usec\nrounds: 3630"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body",
            "value": 43668.96981264564,
            "unit": "iter/sec",
            "range": "stddev: 0.000010891753061156029",
            "extra": "mean: 22.899555549176714 usec\nrounds: 9"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body_vmap",
            "value": 33967.006706031556,
            "unit": "iter/sec",
            "range": "stddev: 0.000010362876347001172",
            "extra": "mean: 29.440333340365516 usec\nrounds: 6"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body",
            "value": 340.7371928679738,
            "unit": "iter/sec",
            "range": "stddev: 0.00010254039647539535",
            "extra": "mean: 2.934813166660888 msec\nrounds: 6"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body_matrix",
            "value": 3.8987476095158846,
            "unit": "iter/sec",
            "range": "stddev: 0.0008901669568816927",
            "extra": "mean: 256.4926228000104 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=1,]",
            "value": 562.8525281703213,
            "unit": "iter/sec",
            "range": "stddev: 0.00004506496587834512",
            "extra": "mean: 1.7766643124989148 msec\nrounds: 112"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.689155537134698,
            "unit": "iter/sec",
            "range": "stddev: 0.0010323558627905274",
            "extra": "mean: 213.2580146000123 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=1,]",
            "value": 1150.5497879069064,
            "unit": "iter/sec",
            "range": "stddev: 0.000019279194729725234",
            "extra": "mean: 869.1496973974604 usec\nrounds: 922"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=10000,]",
            "value": 17.249347864682484,
            "unit": "iter/sec",
            "range": "stddev: 0.0003040619490573995",
            "extra": "mean: 57.97320616667889 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 564.8530811791456,
            "unit": "iter/sec",
            "range": "stddev: 0.00003661000207885276",
            "extra": "mean: 1.7703718600816938 msec\nrounds: 486"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.6785478370273825,
            "unit": "iter/sec",
            "range": "stddev: 0.0019197689224773596",
            "extra": "mean: 213.74153579999984 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1160.5113534199675,
            "unit": "iter/sec",
            "range": "stddev: 0.000022508224577696302",
            "extra": "mean: 861.6891140729052 usec\nrounds: 938"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 17.144850631788856,
            "unit": "iter/sec",
            "range": "stddev: 0.00018881164770573135",
            "extra": "mean: 58.326550722224766 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=1,]",
            "value": 565.4047256574726,
            "unit": "iter/sec",
            "range": "stddev: 0.00003714689269898649",
            "extra": "mean: 1.7686445737381566 msec\nrounds: 495"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.723220867766992,
            "unit": "iter/sec",
            "range": "stddev: 0.00029438622408774044",
            "extra": "mean: 211.71993180000754 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=1,]",
            "value": 1165.9081505687573,
            "unit": "iter/sec",
            "range": "stddev: 0.000014457508531754493",
            "extra": "mean: 857.7004968291684 usec\nrounds: 946"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=10000,]",
            "value": 17.681796185620623,
            "unit": "iter/sec",
            "range": "stddev: 0.0003072973813026703",
            "extra": "mean: 56.555340277772835 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 568.6280527387198,
            "unit": "iter/sec",
            "range": "stddev: 0.000021482881746092744",
            "extra": "mean: 1.7586188285710416 msec\nrounds: 490"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.687562915883644,
            "unit": "iter/sec",
            "range": "stddev: 0.0003045307699030028",
            "extra": "mean: 213.33046999999397 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1144.740015189815,
            "unit": "iter/sec",
            "range": "stddev: 0.000020028065125992563",
            "extra": "mean: 873.5607969764077 usec\nrounds: 926"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 17.47666828132403,
            "unit": "iter/sec",
            "range": "stddev: 0.0003708227799202439",
            "extra": "mean: 57.219144055541925 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/orbits/tests/test_benchmarks.py::test_benchmark_iterate_real_orbits",
            "value": 19837.750006721355,
            "unit": "iter/sec",
            "range": "stddev: 0.000002494701223981751",
            "extra": "mean: 50.40894252932836 usec\nrounds: 14999"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1401.2516203554158,
            "unit": "iter/sec",
            "range": "stddev: 0.000017359167695437558",
            "extra": "mean: 713.6477028632148 usec\nrounds: 1013"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 17.436255570625868,
            "unit": "iter/sec",
            "range": "stddev: 0.00013639752870028265",
            "extra": "mean: 57.35176316666627 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1434.88970508748,
            "unit": "iter/sec",
            "range": "stddev: 0.00001626965322454726",
            "extra": "mean: 696.9176769855169 usec\nrounds: 1260"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 19.311193686394848,
            "unit": "iter/sec",
            "range": "stddev: 0.0012669372467950307",
            "extra": "mean: 51.78343795000728 msec\nrounds: 20"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1389.5979258118903,
            "unit": "iter/sec",
            "range": "stddev: 0.0000175326057149557",
            "extra": "mean: 719.6326228075918 usec\nrounds: 1254"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 18.264285327622545,
            "unit": "iter/sec",
            "range": "stddev: 0.0011643432002441125",
            "extra": "mean: 54.751663263145566 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1387.3230051793025,
            "unit": "iter/sec",
            "range": "stddev: 0.000014747312704136327",
            "extra": "mean: 720.8126703490775 usec\nrounds: 1177"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 17.229953491138346,
            "unit": "iter/sec",
            "range": "stddev: 0.0011269185981877174",
            "extra": "mean: 58.03846194444557 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1415.689448524885,
            "unit": "iter/sec",
            "range": "stddev: 0.000015885179733286005",
            "extra": "mean: 706.3696074318958 usec\nrounds: 1238"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 19.22340441182687,
            "unit": "iter/sec",
            "range": "stddev: 0.0018586618046606516",
            "extra": "mean: 52.01992209999844 msec\nrounds: 20"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1399.781288956998,
            "unit": "iter/sec",
            "range": "stddev: 0.000015954217783318462",
            "extra": "mean: 714.3973189876812 usec\nrounds: 1185"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 18.125005447639243,
            "unit": "iter/sec",
            "range": "stddev: 0.0003490623809612642",
            "extra": "mean: 55.17239721052049 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1397.1689510207373,
            "unit": "iter/sec",
            "range": "stddev: 0.00001539202743987803",
            "extra": "mean: 715.7330538081487 usec\nrounds: 1208"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 17.76247007229642,
            "unit": "iter/sec",
            "range": "stddev: 0.00048047962811395925",
            "extra": "mean: 56.298476277782406 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1398.0151355081168,
            "unit": "iter/sec",
            "range": "stddev: 0.000016263023702231756",
            "extra": "mean: 715.2998380354045 usec\nrounds: 1241"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 18.403185842530267,
            "unit": "iter/sec",
            "range": "stddev: 0.0001668303651061149",
            "extra": "mean: 54.33841773683406 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1407.926091838721,
            "unit": "iter/sec",
            "range": "stddev: 0.000016896391908641663",
            "extra": "mean: 710.2645556444099 usec\nrounds: 1249"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 19.404391384721286,
            "unit": "iter/sec",
            "range": "stddev: 0.0002484257929730384",
            "extra": "mean: 51.53472635000469 msec\nrounds: 20"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1399.7211105615147,
            "unit": "iter/sec",
            "range": "stddev: 0.000016366385391251646",
            "extra": "mean: 714.4280331664343 usec\nrounds: 1206"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 17.65686096064898,
            "unit": "iter/sec",
            "range": "stddev: 0.0003290643759202914",
            "extra": "mean: 56.635208388889346 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1398.9506183409612,
            "unit": "iter/sec",
            "range": "stddev: 0.000015069718935025996",
            "extra": "mean: 714.8215147050126 usec\nrounds: 1224"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 18.098229991204928,
            "unit": "iter/sec",
            "range": "stddev: 0.0002544428092930351",
            "extra": "mean: 55.254022105253554 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1398.0905933682302,
            "unit": "iter/sec",
            "range": "stddev: 0.000045079234435065016",
            "extra": "mean: 715.2612318139096 usec\nrounds: 1251"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 19.349182151044374,
            "unit": "iter/sec",
            "range": "stddev: 0.000644668044471691",
            "extra": "mean: 51.68177094999464 msec\nrounds: 20"
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
          "id": "b49e49a1bb4fff3d070c256d44396f7ac5165e57",
          "message": "Use new serialization assist for unit tests",
          "timestamp": "2025-01-24T11:11:59-05:00",
          "tree_id": "ba03f8e95a220334b6301081721ef4b517e1ad45",
          "url": "https://github.com/B612-Asteroid-Institute/adam_core/commit/b49e49a1bb4fff3d070c256d44396f7ac5165e57"
        },
        "date": 1737739851720,
        "tool": "pytest",
        "benches": [
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 427.308198709838,
            "unit": "iter/sec",
            "range": "stddev: 0.00008081952893744933",
            "extra": "mean: 2.3402312499953837 msec\nrounds: 8"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 293.5537131070101,
            "unit": "iter/sec",
            "range": "stddev: 0.00020601906167289443",
            "extra": "mean: 3.4065316000123858 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 313.94767157938196,
            "unit": "iter/sec",
            "range": "stddev: 0.00021098291580505171",
            "extra": "mean: 3.185244199994486 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 444.23885003400613,
            "unit": "iter/sec",
            "range": "stddev: 0.0000415340482056316",
            "extra": "mean: 2.251041303396699 msec\nrounds: 412"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 306.6592277953729,
            "unit": "iter/sec",
            "range": "stddev: 0.000026935371780115187",
            "extra": "mean: 3.2609486666655227 msec\nrounds: 285"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 326.7219960153299,
            "unit": "iter/sec",
            "range": "stddev: 0.00003321041256327364",
            "extra": "mean: 3.0607060809982305 msec\nrounds: 321"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 420.00539355693235,
            "unit": "iter/sec",
            "range": "stddev: 0.0000393170968234094",
            "extra": "mean: 2.380921805625452 msec\nrounds: 391"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 259.85473335927605,
            "unit": "iter/sec",
            "range": "stddev: 0.000045817792338161755",
            "extra": "mean: 3.8483039622657036 msec\nrounds: 53"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 262.15410666229207,
            "unit": "iter/sec",
            "range": "stddev: 0.00018548485264892534",
            "extra": "mean: 3.8145502000020315 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 421.36385112844755,
            "unit": "iter/sec",
            "range": "stddev: 0.000031181561951971445",
            "extra": "mean: 2.3732458238216605 msec\nrounds: 403"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 258.4362682959845,
            "unit": "iter/sec",
            "range": "stddev: 0.00012810362952109816",
            "extra": "mean: 3.8694259385246568 msec\nrounds: 244"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 271.0893951016419,
            "unit": "iter/sec",
            "range": "stddev: 0.00002339496289041614",
            "extra": "mean: 3.688820064779964 msec\nrounds: 247"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 405.7068196148007,
            "unit": "iter/sec",
            "range": "stddev: 0.00004034086729131176",
            "extra": "mean: 2.46483409115344 msec\nrounds: 373"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 225.7000665088491,
            "unit": "iter/sec",
            "range": "stddev: 0.00029759192456995997",
            "extra": "mean: 4.430658862746913 msec\nrounds: 51"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 231.78400252677946,
            "unit": "iter/sec",
            "range": "stddev: 0.00019929550008411433",
            "extra": "mean: 4.314361600017946 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 407.588599756255,
            "unit": "iter/sec",
            "range": "stddev: 0.00007799666139256415",
            "extra": "mean: 2.4534542933684045 msec\nrounds: 392"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 226.23958560599664,
            "unit": "iter/sec",
            "range": "stddev: 0.00009751661160618403",
            "extra": "mean: 4.420092961722142 msec\nrounds: 209"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 234.57417313182063,
            "unit": "iter/sec",
            "range": "stddev: 0.00004426855793954511",
            "extra": "mean: 4.263043909092426 msec\nrounds: 231"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_to_matrix",
            "value": 660.0946767977332,
            "unit": "iter/sec",
            "range": "stddev: 0.000028534641304228297",
            "extra": "mean: 1.5149341983050424 msec\nrounds: 590"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_from_matrix",
            "value": 10720.852376945973,
            "unit": "iter/sec",
            "range": "stddev: 0.000006853024890241868",
            "extra": "mean: 93.27616544281416 usec\nrounds: 2859"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body",
            "value": 37912.13651076376,
            "unit": "iter/sec",
            "range": "stddev: 0.000021104342160582016",
            "extra": "mean: 26.37677778238867 usec\nrounds: 9"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body_vmap",
            "value": 33004.23396203812,
            "unit": "iter/sec",
            "range": "stddev: 0.000010439489634202486",
            "extra": "mean: 30.29914286603993 usec\nrounds: 7"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body",
            "value": 332.6004150583982,
            "unit": "iter/sec",
            "range": "stddev: 0.0001281308293270287",
            "extra": "mean: 3.006610800002818 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body_matrix",
            "value": 3.857613162019349,
            "unit": "iter/sec",
            "range": "stddev: 0.007597295844986296",
            "extra": "mean: 259.2276513999991 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=1,]",
            "value": 547.8301548336079,
            "unit": "iter/sec",
            "range": "stddev: 0.000034215391056474986",
            "extra": "mean: 1.8253832710317477 msec\nrounds: 107"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.780966008946182,
            "unit": "iter/sec",
            "range": "stddev: 0.0023266830505595886",
            "extra": "mean: 209.16275039998027 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=1,]",
            "value": 1128.4596013429534,
            "unit": "iter/sec",
            "range": "stddev: 0.000013522432077927382",
            "extra": "mean: 886.163757045377 usec\nrounds: 1029"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=10000,]",
            "value": 17.324271568953023,
            "unit": "iter/sec",
            "range": "stddev: 0.0009168028129781048",
            "extra": "mean: 57.7224846666632 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 550.3856347374096,
            "unit": "iter/sec",
            "range": "stddev: 0.00006926583596277427",
            "extra": "mean: 1.8169078858264578 msec\nrounds: 508"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.785136655052485,
            "unit": "iter/sec",
            "range": "stddev: 0.000249984875825603",
            "extra": "mean: 208.98044759999266 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1124.6686228462754,
            "unit": "iter/sec",
            "range": "stddev: 0.00004046894756892088",
            "extra": "mean: 889.1507948974623 usec\nrounds: 980"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 17.225199502259606,
            "unit": "iter/sec",
            "range": "stddev: 0.0004349840201573746",
            "extra": "mean: 58.05448000000347 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=1,]",
            "value": 558.3755244925908,
            "unit": "iter/sec",
            "range": "stddev: 0.000019059146616828248",
            "extra": "mean: 1.7909094437989987 msec\nrounds: 516"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.832527236700784,
            "unit": "iter/sec",
            "range": "stddev: 0.0007855618290844764",
            "extra": "mean: 206.93106340000895 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=1,]",
            "value": 1137.1859491392406,
            "unit": "iter/sec",
            "range": "stddev: 0.000015759125311027782",
            "extra": "mean: 879.3636614635633 usec\nrounds: 1025"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=10000,]",
            "value": 17.707354588004563,
            "unit": "iter/sec",
            "range": "stddev: 0.00018859184889344417",
            "extra": "mean: 56.47370955554405 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 556.91540674564,
            "unit": "iter/sec",
            "range": "stddev: 0.00003757428520694894",
            "extra": "mean: 1.7956048403177507 msec\nrounds: 501"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.788653977796409,
            "unit": "iter/sec",
            "range": "stddev: 0.0018465443903980978",
            "extra": "mean: 208.82694900001297 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1138.3345600835094,
            "unit": "iter/sec",
            "range": "stddev: 0.00001322405182411081",
            "extra": "mean: 878.4763592933864 usec\nrounds: 963"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 17.548375619255566,
            "unit": "iter/sec",
            "range": "stddev: 0.00016324550385471255",
            "extra": "mean: 56.985331388890216 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/orbits/tests/test_benchmarks.py::test_benchmark_iterate_real_orbits",
            "value": 20110.963960252233,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024982462052524233",
            "extra": "mean: 49.72412073217488 usec\nrounds: 17104"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1384.9987971260073,
            "unit": "iter/sec",
            "range": "stddev: 0.00003337249991599149",
            "extra": "mean: 722.0222877269546 usec\nrounds: 994"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 17.429483268865482,
            "unit": "iter/sec",
            "range": "stddev: 0.0002248614690925738",
            "extra": "mean: 57.37404744444222 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1404.7628931260158,
            "unit": "iter/sec",
            "range": "stddev: 0.000014588181311703229",
            "extra": "mean: 711.8639059255775 usec\nrounds: 1350"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 19.186268874497607,
            "unit": "iter/sec",
            "range": "stddev: 0.0008596435186278782",
            "extra": "mean: 52.12060805262665 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1388.6686642727393,
            "unit": "iter/sec",
            "range": "stddev: 0.000018186540166452866",
            "extra": "mean: 720.1141825459932 usec\nrounds: 1249"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 18.116325757222707,
            "unit": "iter/sec",
            "range": "stddev: 0.0019269440325562267",
            "extra": "mean: 55.1988307894781 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1383.6678418178049,
            "unit": "iter/sec",
            "range": "stddev: 0.000026075053941526586",
            "extra": "mean: 722.7168036847933 usec\nrounds: 1248"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 17.207376808553857,
            "unit": "iter/sec",
            "range": "stddev: 0.0007443306723666599",
            "extra": "mean: 58.1146104444517 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1385.7484742325705,
            "unit": "iter/sec",
            "range": "stddev: 0.000017489996739215576",
            "extra": "mean: 721.6316803479084 usec\nrounds: 1267"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 19.166171511021517,
            "unit": "iter/sec",
            "range": "stddev: 0.0011153574294624237",
            "extra": "mean: 52.175260949999824 msec\nrounds: 20"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1377.8836814056892,
            "unit": "iter/sec",
            "range": "stddev: 0.000026280097021780725",
            "extra": "mean: 725.750666398647 usec\nrounds: 1244"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 18.115605697886828,
            "unit": "iter/sec",
            "range": "stddev: 0.0003158106697650341",
            "extra": "mean: 55.20102483333744 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1376.9267541521372,
            "unit": "iter/sec",
            "range": "stddev: 0.000014263549115227363",
            "extra": "mean: 726.2550436938562 usec\nrounds: 1213"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 17.65900053219574,
            "unit": "iter/sec",
            "range": "stddev: 0.00024804684952456964",
            "extra": "mean: 56.62834644445525 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1387.6772079364748,
            "unit": "iter/sec",
            "range": "stddev: 0.000013955215609754562",
            "extra": "mean: 720.6286838760113 usec\nrounds: 1259"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 18.111365701012605,
            "unit": "iter/sec",
            "range": "stddev: 0.0008262683766709116",
            "extra": "mean: 55.21394777778078 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1413.2575162453652,
            "unit": "iter/sec",
            "range": "stddev: 0.00001354435981721892",
            "extra": "mean: 707.585127625377 usec\nrounds: 1285"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 19.234294710538304,
            "unit": "iter/sec",
            "range": "stddev: 0.00020748961956670699",
            "extra": "mean: 51.990468850002 msec\nrounds: 20"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1369.1894185841713,
            "unit": "iter/sec",
            "range": "stddev: 0.000029178698038717622",
            "extra": "mean: 730.3591354321621 usec\nrounds: 1270"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 17.5660039940785,
            "unit": "iter/sec",
            "range": "stddev: 0.00030467035160410244",
            "extra": "mean: 56.92814372222048 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1395.9073758321945,
            "unit": "iter/sec",
            "range": "stddev: 0.000012247252839950048",
            "extra": "mean: 716.3799098087238 usec\nrounds: 1264"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 18.103093241946677,
            "unit": "iter/sec",
            "range": "stddev: 0.00017312867136579544",
            "extra": "mean: 55.23917855556862 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1407.0266284706893,
            "unit": "iter/sec",
            "range": "stddev: 0.0000248979279656517",
            "extra": "mean: 710.7186031631182 usec\nrounds: 1328"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 19.253579868378466,
            "unit": "iter/sec",
            "range": "stddev: 0.0001669961466037472",
            "extra": "mean: 51.938393111110294 msec\nrounds: 18"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "akoumjian@users.noreply.github.com",
            "name": "Alec Koumjian",
            "username": "akoumjian"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "6d0458e55bba444a67a106d79088793b065871e2",
          "message": "Disable sbdb cache (#142)\n\n* Disable sbdb cache",
          "timestamp": "2025-02-05T15:24:30-05:00",
          "tree_id": "33c9e28cf7faec08b923d5aa7687bc28f7c23e85",
          "url": "https://github.com/B612-Asteroid-Institute/adam_core/commit/6d0458e55bba444a67a106d79088793b065871e2"
        },
        "date": 1738787308482,
        "tool": "pytest",
        "benches": [
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 405.5626039764906,
            "unit": "iter/sec",
            "range": "stddev: 0.00005083055398429701",
            "extra": "mean: 2.4657105714262735 msec\nrounds: 7"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 289.72374782781753,
            "unit": "iter/sec",
            "range": "stddev: 0.0002356977920567628",
            "extra": "mean: 3.4515637999902538 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 305.1358513632968,
            "unit": "iter/sec",
            "range": "stddev: 0.00014018506598476165",
            "extra": "mean: 3.2772287999989658 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 427.69954429701005,
            "unit": "iter/sec",
            "range": "stddev: 0.00009111469114670577",
            "extra": "mean: 2.338089935643148 msec\nrounds: 404"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 300.7720544307118,
            "unit": "iter/sec",
            "range": "stddev: 0.00011901071509815043",
            "extra": "mean: 3.324776970695486 msec\nrounds: 273"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 323.07201354973023,
            "unit": "iter/sec",
            "range": "stddev: 0.00007695469226644181",
            "extra": "mean: 3.095285131672573 msec\nrounds: 281"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 408.50926948166756,
            "unit": "iter/sec",
            "range": "stddev: 0.00010582983152614328",
            "extra": "mean: 2.447924869046029 msec\nrounds: 336"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 253.95030070521375,
            "unit": "iter/sec",
            "range": "stddev: 0.00013565756656316668",
            "extra": "mean: 3.9377783653849767 msec\nrounds: 52"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 258.17134243407725,
            "unit": "iter/sec",
            "range": "stddev: 0.0001870011692944698",
            "extra": "mean: 3.873396599993839 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 416.96414252514484,
            "unit": "iter/sec",
            "range": "stddev: 0.00007456845977950557",
            "extra": "mean: 2.3982877614942524 msec\nrounds: 348"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 255.91078645789366,
            "unit": "iter/sec",
            "range": "stddev: 0.00014506504849701838",
            "extra": "mean: 3.9076117651826108 msec\nrounds: 247"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 263.1248149758543,
            "unit": "iter/sec",
            "range": "stddev: 0.00014035704082807345",
            "extra": "mean: 3.8004777318010285 msec\nrounds: 261"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 396.63757390160646,
            "unit": "iter/sec",
            "range": "stddev: 0.00008623594293689937",
            "extra": "mean: 2.521193315507898 msec\nrounds: 374"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 219.6078555972743,
            "unit": "iter/sec",
            "range": "stddev: 0.0002670767744342798",
            "extra": "mean: 4.553571170212782 msec\nrounds: 47"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 226.9687233015744,
            "unit": "iter/sec",
            "range": "stddev: 0.0002074208816772932",
            "extra": "mean: 4.40589339999633 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 394.2328343605687,
            "unit": "iter/sec",
            "range": "stddev: 0.00023569305670538218",
            "extra": "mean: 2.5365720783302175 msec\nrounds: 383"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 221.0154768134912,
            "unit": "iter/sec",
            "range": "stddev: 0.00021081677759821692",
            "extra": "mean: 4.5245700184330175 msec\nrounds: 217"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 227.8147655217529,
            "unit": "iter/sec",
            "range": "stddev: 0.00015145644050078427",
            "extra": "mean: 4.389531107475626 msec\nrounds: 214"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_to_matrix",
            "value": 687.2943895424374,
            "unit": "iter/sec",
            "range": "stddev: 0.00010002210138393126",
            "extra": "mean: 1.4549805952377186 msec\nrounds: 588"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_from_matrix",
            "value": 10680.717357004574,
            "unit": "iter/sec",
            "range": "stddev: 0.000011341693464812692",
            "extra": "mean: 93.62667006108771 usec\nrounds: 2946"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body",
            "value": 45219.31367027416,
            "unit": "iter/sec",
            "range": "stddev: 0.000009559782118584037",
            "extra": "mean: 22.114444444948983 usec\nrounds: 9"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body_vmap",
            "value": 32251.673636882904,
            "unit": "iter/sec",
            "range": "stddev: 0.000012496852682747005",
            "extra": "mean: 31.006142851960508 usec\nrounds: 7"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body",
            "value": 316.5345794245251,
            "unit": "iter/sec",
            "range": "stddev: 0.0002036535774531681",
            "extra": "mean: 3.1592124999993607 msec\nrounds: 6"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body_matrix",
            "value": 3.8843697156209256,
            "unit": "iter/sec",
            "range": "stddev: 0.0005683290033355374",
            "extra": "mean: 257.44202359999804 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=1,]",
            "value": 557.5956911761918,
            "unit": "iter/sec",
            "range": "stddev: 0.00005775797171476139",
            "extra": "mean: 1.7934141454547483 msec\nrounds: 110"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.6827633146656575,
            "unit": "iter/sec",
            "range": "stddev: 0.0032788253487806684",
            "extra": "mean: 213.54912320000494 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=1,]",
            "value": 1148.1049533921205,
            "unit": "iter/sec",
            "range": "stddev: 0.00002889872144907885",
            "extra": "mean: 871.000510053947 usec\nrounds: 945"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=10000,]",
            "value": 17.27303925260442,
            "unit": "iter/sec",
            "range": "stddev: 0.0004427392181256647",
            "extra": "mean: 57.893691166667175 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 561.1261667492972,
            "unit": "iter/sec",
            "range": "stddev: 0.00007840006873061255",
            "extra": "mean: 1.782130400001084 msec\nrounds: 490"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.68534793916209,
            "unit": "iter/sec",
            "range": "stddev: 0.0009716591668325623",
            "extra": "mean: 213.43132100000162 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1152.3216854785414,
            "unit": "iter/sec",
            "range": "stddev: 0.000029199682682733385",
            "extra": "mean: 867.8132266379379 usec\nrounds: 931"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 17.09471648383752,
            "unit": "iter/sec",
            "range": "stddev: 0.0004887931393052618",
            "extra": "mean: 58.49760661111089 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=1,]",
            "value": 564.0960928297391,
            "unit": "iter/sec",
            "range": "stddev: 0.00004733466634897186",
            "extra": "mean: 1.7727476093365349 msec\nrounds: 407"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.714840732269925,
            "unit": "iter/sec",
            "range": "stddev: 0.0011547479556430144",
            "extra": "mean: 212.09624179999764 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=1,]",
            "value": 1160.0783708060828,
            "unit": "iter/sec",
            "range": "stddev: 0.000022583651536841646",
            "extra": "mean: 862.0107271762579 usec\nrounds: 953"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=10000,]",
            "value": 17.544495286529205,
            "unit": "iter/sec",
            "range": "stddev: 0.0006757825881790894",
            "extra": "mean: 56.997934888888345 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 562.5627326385868,
            "unit": "iter/sec",
            "range": "stddev: 0.000048484641042189226",
            "extra": "mean: 1.7775795337698643 msec\nrounds: 459"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.704473167299861,
            "unit": "iter/sec",
            "range": "stddev: 0.0004850045245211469",
            "extra": "mean: 212.56365260001076 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1117.4383213893555,
            "unit": "iter/sec",
            "range": "stddev: 0.00008082403513304266",
            "extra": "mean: 894.903978911928 usec\nrounds: 901"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 17.292217618536476,
            "unit": "iter/sec",
            "range": "stddev: 0.0002348919753601659",
            "extra": "mean: 57.82948272221865 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/orbits/tests/test_benchmarks.py::test_benchmark_iterate_real_orbits",
            "value": 19384.76745210954,
            "unit": "iter/sec",
            "range": "stddev: 0.000004496786271817304",
            "extra": "mean: 51.58689690090533 usec\nrounds: 15907"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1372.485589618743,
            "unit": "iter/sec",
            "range": "stddev: 0.00002171758793382375",
            "extra": "mean: 728.6050998012925 usec\nrounds: 1012"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 17.246126835374533,
            "unit": "iter/sec",
            "range": "stddev: 0.0005232022411823138",
            "extra": "mean: 57.984033722217674 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1396.2019351481038,
            "unit": "iter/sec",
            "range": "stddev: 0.000036657511028330267",
            "extra": "mean: 716.2287738083702 usec\nrounds: 1260"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 19.139022370413617,
            "unit": "iter/sec",
            "range": "stddev: 0.0006311478809535552",
            "extra": "mean: 52.24927274999516 msec\nrounds: 20"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1418.9085747731517,
            "unit": "iter/sec",
            "range": "stddev: 0.000028362664560050538",
            "extra": "mean: 704.7670426263194 usec\nrounds: 1173"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 18.2179183672064,
            "unit": "iter/sec",
            "range": "stddev: 0.00046171007220679203",
            "extra": "mean: 54.891013333338556 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1396.0934960222635,
            "unit": "iter/sec",
            "range": "stddev: 0.000020096685976836253",
            "extra": "mean: 716.2844056284127 usec\nrounds: 1208"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 17.137738296284763,
            "unit": "iter/sec",
            "range": "stddev: 0.0008984082613264925",
            "extra": "mean: 58.35075683334404 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1418.8564930946839,
            "unit": "iter/sec",
            "range": "stddev: 0.000023071669396230344",
            "extra": "mean: 704.7929123676834 usec\nrounds: 1221"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 19.115059599379624,
            "unit": "iter/sec",
            "range": "stddev: 0.0005573841154307926",
            "extra": "mean: 52.31477279999979 msec\nrounds: 20"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1403.2125140085693,
            "unit": "iter/sec",
            "range": "stddev: 0.00002247304604445768",
            "extra": "mean: 712.6504289384444 usec\nrounds: 1168"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 17.87459282199053,
            "unit": "iter/sec",
            "range": "stddev: 0.0016306520098220646",
            "extra": "mean: 55.94533033332835 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1392.3308923625978,
            "unit": "iter/sec",
            "range": "stddev: 0.000028330629073194562",
            "extra": "mean: 718.2200764813418 usec\nrounds: 1046"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 17.4835565281121,
            "unit": "iter/sec",
            "range": "stddev: 0.001540510325011281",
            "extra": "mean: 57.19660061109898 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1400.7767585411782,
            "unit": "iter/sec",
            "range": "stddev: 0.00002733966920558117",
            "extra": "mean: 713.8896286667675 usec\nrounds: 1193"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 18.213146414539214,
            "unit": "iter/sec",
            "range": "stddev: 0.0005743840141140136",
            "extra": "mean: 54.90539510524765 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1413.0595213493434,
            "unit": "iter/sec",
            "range": "stddev: 0.00002006948370164108",
            "extra": "mean: 707.6842729491613 usec\nrounds: 1231"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 19.151968383196536,
            "unit": "iter/sec",
            "range": "stddev: 0.0004081829532344548",
            "extra": "mean: 52.21395419999624 msec\nrounds: 20"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1386.4142094568888,
            "unit": "iter/sec",
            "range": "stddev: 0.00002284166073450431",
            "extra": "mean: 721.285163682604 usec\nrounds: 1173"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 17.464451528544032,
            "unit": "iter/sec",
            "range": "stddev: 0.0007842256863425873",
            "extra": "mean: 57.25917005556072 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1381.4990568846542,
            "unit": "iter/sec",
            "range": "stddev: 0.00003137505779377923",
            "extra": "mean: 723.8513808724903 usec\nrounds: 1192"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 18.002406111586012,
            "unit": "iter/sec",
            "range": "stddev: 0.0004569326014255992",
            "extra": "mean: 55.54813027778652 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1407.7393848667339,
            "unit": "iter/sec",
            "range": "stddev: 0.00003255188637728458",
            "extra": "mean: 710.3587572742854 usec\nrounds: 1203"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 19.122924175982373,
            "unit": "iter/sec",
            "range": "stddev: 0.0013265500158696777",
            "extra": "mean: 52.293257600004495 msec\nrounds: 20"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "akoumjian@users.noreply.github.com",
            "name": "Alec Koumjian",
            "username": "akoumjian"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "6d0458e55bba444a67a106d79088793b065871e2",
          "message": "Disable sbdb cache (#142)\n\n* Disable sbdb cache",
          "timestamp": "2025-02-05T15:24:30-05:00",
          "tree_id": "33c9e28cf7faec08b923d5aa7687bc28f7c23e85",
          "url": "https://github.com/B612-Asteroid-Institute/adam_core/commit/6d0458e55bba444a67a106d79088793b065871e2"
        },
        "date": 1738787326568,
        "tool": "pytest",
        "benches": [
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 420.58387977704035,
            "unit": "iter/sec",
            "range": "stddev: 0.00009155229939636295",
            "extra": "mean: 2.3776469999994276 msec\nrounds: 8"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 292.04222323097184,
            "unit": "iter/sec",
            "range": "stddev: 0.00026681501462808374",
            "extra": "mean: 3.424162400000341 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 306.70987350002827,
            "unit": "iter/sec",
            "range": "stddev: 0.00027068093377716153",
            "extra": "mean: 3.260410199999342 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 433.3447258987601,
            "unit": "iter/sec",
            "range": "stddev: 0.00008523354170657906",
            "extra": "mean: 2.3076316388205553 msec\nrounds: 407"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 305.26080553722034,
            "unit": "iter/sec",
            "range": "stddev: 0.00011123199108673977",
            "extra": "mean: 3.275887312949092 msec\nrounds: 278"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 325.5053563107029,
            "unit": "iter/sec",
            "range": "stddev: 0.000023476375124616808",
            "extra": "mean: 3.0721460664551254 msec\nrounds: 316"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 413.6992814880679,
            "unit": "iter/sec",
            "range": "stddev: 0.0000875036884563599",
            "extra": "mean: 2.417214737243488 msec\nrounds: 392"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 259.68148943499,
            "unit": "iter/sec",
            "range": "stddev: 0.00006861469817264267",
            "extra": "mean: 3.850871319999669 msec\nrounds: 50"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 260.70599600814137,
            "unit": "iter/sec",
            "range": "stddev: 0.00020244936887939017",
            "extra": "mean: 3.8357384000050843 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 417.48352172561073,
            "unit": "iter/sec",
            "range": "stddev: 0.00008961740450957252",
            "extra": "mean: 2.3953041209067067 msec\nrounds: 397"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 257.7261918782024,
            "unit": "iter/sec",
            "range": "stddev: 0.00014732123959633462",
            "extra": "mean: 3.880086818931407 msec\nrounds: 243"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 270.07645272084034,
            "unit": "iter/sec",
            "range": "stddev: 0.00012345828763892923",
            "extra": "mean: 3.702655266409441 msec\nrounds: 259"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 400.13683733278197,
            "unit": "iter/sec",
            "range": "stddev: 0.00006331964638836983",
            "extra": "mean: 2.49914505913968 msec\nrounds: 372"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 220.79826540887672,
            "unit": "iter/sec",
            "range": "stddev: 0.00010926060578825023",
            "extra": "mean: 4.529021086955502 msec\nrounds: 46"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 220.987530513511,
            "unit": "iter/sec",
            "range": "stddev: 0.0002199655808962559",
            "extra": "mean: 4.525142199997845 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 403.7435898221339,
            "unit": "iter/sec",
            "range": "stddev: 0.00009167040886511011",
            "extra": "mean: 2.4768195092349137 msec\nrounds: 379"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 225.58457156981336,
            "unit": "iter/sec",
            "range": "stddev: 0.00006627430863343902",
            "extra": "mean: 4.432927274419219 msec\nrounds: 215"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 227.93220865192959,
            "unit": "iter/sec",
            "range": "stddev: 0.00015404634849202687",
            "extra": "mean: 4.387269381165339 msec\nrounds: 223"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_to_matrix",
            "value": 647.3652005137963,
            "unit": "iter/sec",
            "range": "stddev: 0.00002855383428062977",
            "extra": "mean: 1.544723131867958 msec\nrounds: 546"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_from_matrix",
            "value": 10502.843143511605,
            "unit": "iter/sec",
            "range": "stddev: 0.000008645411155751885",
            "extra": "mean: 95.21231406924086 usec\nrounds: 3426"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body",
            "value": 43400.68477861986,
            "unit": "iter/sec",
            "range": "stddev: 0.000009646798342664943",
            "extra": "mean: 23.041111104602248 usec\nrounds: 9"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body_vmap",
            "value": 34897.57560658148,
            "unit": "iter/sec",
            "range": "stddev: 0.000011627407811244835",
            "extra": "mean: 28.65528572166503 usec\nrounds: 7"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body",
            "value": 329.6132631099117,
            "unit": "iter/sec",
            "range": "stddev: 0.00010928736958506279",
            "extra": "mean: 3.0338585000038165 msec\nrounds: 6"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body_matrix",
            "value": 3.7800503780015857,
            "unit": "iter/sec",
            "range": "stddev: 0.0015396448580708257",
            "extra": "mean: 264.5467387999929 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=1,]",
            "value": 551.550097417835,
            "unit": "iter/sec",
            "range": "stddev: 0.000028927613438932185",
            "extra": "mean: 1.8130719306943301 msec\nrounds: 101"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.710475130478006,
            "unit": "iter/sec",
            "range": "stddev: 0.0004326294462180632",
            "extra": "mean: 212.29280959997823 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=1,]",
            "value": 1108.407683777016,
            "unit": "iter/sec",
            "range": "stddev: 0.000022062668999030313",
            "extra": "mean: 902.1951170460986 usec\nrounds: 880"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=10000,]",
            "value": 16.825605932289452,
            "unit": "iter/sec",
            "range": "stddev: 0.0003189515990652091",
            "extra": "mean: 59.433223625006804 msec\nrounds: 16"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 552.8911456759497,
            "unit": "iter/sec",
            "range": "stddev: 0.000026262539988711915",
            "extra": "mean: 1.8086742893620176 msec\nrounds: 470"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.649317924046375,
            "unit": "iter/sec",
            "range": "stddev: 0.0011494594812824083",
            "extra": "mean: 215.08531279996532 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1100.9224585458162,
            "unit": "iter/sec",
            "range": "stddev: 0.0000361744300331602",
            "extra": "mean: 908.3291854367997 usec\nrounds: 879"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 16.62559751260513,
            "unit": "iter/sec",
            "range": "stddev: 0.0011079197882222704",
            "extra": "mean: 60.1482141764724 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=1,]",
            "value": 539.6334189695658,
            "unit": "iter/sec",
            "range": "stddev: 0.00003589729836850357",
            "extra": "mean: 1.8531098424362 msec\nrounds: 476"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.706454449610551,
            "unit": "iter/sec",
            "range": "stddev: 0.0006333176634008841",
            "extra": "mean: 212.47416940001358 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=1,]",
            "value": 1110.8719180734022,
            "unit": "iter/sec",
            "range": "stddev: 0.00001960128858654359",
            "extra": "mean: 900.1937880780274 usec\nrounds: 906"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=10000,]",
            "value": 16.86653823697507,
            "unit": "iter/sec",
            "range": "stddev: 0.0006392791010295592",
            "extra": "mean: 59.28898899999441 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 545.9254976107793,
            "unit": "iter/sec",
            "range": "stddev: 0.00005085040975857037",
            "extra": "mean: 1.8317517763439501 msec\nrounds: 465"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.694920824132399,
            "unit": "iter/sec",
            "range": "stddev: 0.0009378415718633763",
            "extra": "mean: 212.99613719999115 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1105.7667946071942,
            "unit": "iter/sec",
            "range": "stddev: 0.000026396114723993647",
            "extra": "mean: 904.3498184942638 usec\nrounds: 876"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 16.809873655438302,
            "unit": "iter/sec",
            "range": "stddev: 0.000609872019673012",
            "extra": "mean: 59.488846882348916 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/orbits/tests/test_benchmarks.py::test_benchmark_iterate_real_orbits",
            "value": 19699.7555166789,
            "unit": "iter/sec",
            "range": "stddev: 0.000002830291911393507",
            "extra": "mean: 50.76205129314142 usec\nrounds: 16669"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1350.9516522621357,
            "unit": "iter/sec",
            "range": "stddev: 0.00002142345587124272",
            "extra": "mean: 740.218939978736 usec\nrounds: 933"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 16.798418839535813,
            "unit": "iter/sec",
            "range": "stddev: 0.0006701402714472395",
            "extra": "mean: 59.529412235302544 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1348.262451394617,
            "unit": "iter/sec",
            "range": "stddev: 0.000030457320284296958",
            "extra": "mean: 741.6953568391814 usec\nrounds: 1177"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 19.050018352075735,
            "unit": "iter/sec",
            "range": "stddev: 0.00027348108789849876",
            "extra": "mean: 52.4933877499933 msec\nrounds: 20"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1364.5024130547367,
            "unit": "iter/sec",
            "range": "stddev: 0.000017743615933296627",
            "extra": "mean: 732.8678868081159 usec\nrounds: 1175"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 17.959516770786138,
            "unit": "iter/sec",
            "range": "stddev: 0.00023416444368126315",
            "extra": "mean: 55.68078544444196 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1355.3352628353716,
            "unit": "iter/sec",
            "range": "stddev: 0.000031187198978352185",
            "extra": "mean: 737.8248226995825 usec\nrounds: 1207"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 16.912383671329536,
            "unit": "iter/sec",
            "range": "stddev: 0.00040945561496799913",
            "extra": "mean: 59.128270705875416 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1374.0123573227706,
            "unit": "iter/sec",
            "range": "stddev: 0.000015334278455345393",
            "extra": "mean: 727.795492282526 usec\nrounds: 1231"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 19.010136995869278,
            "unit": "iter/sec",
            "range": "stddev: 0.0009990742760345014",
            "extra": "mean: 52.60351359999618 msec\nrounds: 20"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1364.903981996224,
            "unit": "iter/sec",
            "range": "stddev: 0.000015936846498692354",
            "extra": "mean: 732.6522694566851 usec\nrounds: 1195"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 17.801714164836923,
            "unit": "iter/sec",
            "range": "stddev: 0.000930754788537915",
            "extra": "mean: 56.17436561110859 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1367.7887757157844,
            "unit": "iter/sec",
            "range": "stddev: 0.000028425891937097415",
            "extra": "mean: 731.1070376905856 usec\nrounds: 1247"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 17.366196954649137,
            "unit": "iter/sec",
            "range": "stddev: 0.0011651698160893135",
            "extra": "mean: 57.583131333327884 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1363.245765444217,
            "unit": "iter/sec",
            "range": "stddev: 0.00002070151468529282",
            "extra": "mean: 733.5434485462329 usec\nrounds: 1273"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 17.803333834890978,
            "unit": "iter/sec",
            "range": "stddev: 0.0010857669633407092",
            "extra": "mean: 56.169255111096085 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1393.835010400221,
            "unit": "iter/sec",
            "range": "stddev: 0.00001569194215941374",
            "extra": "mean: 717.4450293889973 usec\nrounds: 1293"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 18.971192130924802,
            "unit": "iter/sec",
            "range": "stddev: 0.0005945922951540031",
            "extra": "mean: 52.71150031578181 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1375.5163986790963,
            "unit": "iter/sec",
            "range": "stddev: 0.00002836620714035645",
            "extra": "mean: 726.9996933226653 usec\nrounds: 1213"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 17.276537096433948,
            "unit": "iter/sec",
            "range": "stddev: 0.0004276947810269929",
            "extra": "mean: 57.88196988888532 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1382.7375482805187,
            "unit": "iter/sec",
            "range": "stddev: 0.000019880940229712924",
            "extra": "mean: 723.2030411291963 usec\nrounds: 1240"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 17.814959948124205,
            "unit": "iter/sec",
            "range": "stddev: 0.0002298289935362176",
            "extra": "mean: 56.132598833335756 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1393.2620447688253,
            "unit": "iter/sec",
            "range": "stddev: 0.000032166815848214856",
            "extra": "mean: 717.740071765124 usec\nrounds: 1059"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 18.970533942271206,
            "unit": "iter/sec",
            "range": "stddev: 0.0006049846792536756",
            "extra": "mean: 52.71332915789703 msec\nrounds: 19"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "moeyensj@users.noreply.github.com",
            "name": "Joachim Moeyens",
            "username": "moeyensj"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "76b8c629c5094eeb89a3282ebccb9b9cce8237b2",
          "message": "Add query_neocc (#141)\n\n* Add query_neocc\n\n* Remove try/except in query_neocc\n\n* Fix linting issues due to updated black (or ruff) versions\n\n* Fix timescale declaration in query_neocc",
          "timestamp": "2025-02-05T15:25:05-05:00",
          "tree_id": "ea6de0b56a4e1ea1c565180796c471f261791b9f",
          "url": "https://github.com/B612-Asteroid-Institute/adam_core/commit/76b8c629c5094eeb89a3282ebccb9b9cce8237b2"
        },
        "date": 1738787357490,
        "tool": "pytest",
        "benches": [
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 431.5199426944871,
            "unit": "iter/sec",
            "range": "stddev: 0.00008935043503592534",
            "extra": "mean: 2.3173899999981984 msec\nrounds: 8"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 296.3213716686086,
            "unit": "iter/sec",
            "range": "stddev: 0.00018683294203667493",
            "extra": "mean: 3.374714400007406 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 314.6752912867276,
            "unit": "iter/sec",
            "range": "stddev: 0.00017867903635061935",
            "extra": "mean: 3.1778790000032586 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 439.57429729857483,
            "unit": "iter/sec",
            "range": "stddev: 0.00007192384626440749",
            "extra": "mean: 2.27492827980514 msec\nrounds: 411"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 307.03679034264314,
            "unit": "iter/sec",
            "range": "stddev: 0.000042849292157475454",
            "extra": "mean: 3.256938684396851 msec\nrounds: 282"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 330.69441273738795,
            "unit": "iter/sec",
            "range": "stddev: 0.00009532946011804255",
            "extra": "mean: 3.023939811145594 msec\nrounds: 323"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 419.74556949078925,
            "unit": "iter/sec",
            "range": "stddev: 0.00009272911335187552",
            "extra": "mean: 2.3823956050641377 msec\nrounds: 395"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 259.1446094104342,
            "unit": "iter/sec",
            "range": "stddev: 0.00013516031614981268",
            "extra": "mean: 3.8588493207520136 msec\nrounds: 53"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 265.812748528274,
            "unit": "iter/sec",
            "range": "stddev: 0.00019369987438688064",
            "extra": "mean: 3.7620468000000074 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 420.19852563998484,
            "unit": "iter/sec",
            "range": "stddev: 0.00006985618605980744",
            "extra": "mean: 2.3798274838707405 msec\nrounds: 403"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 262.7918662087303,
            "unit": "iter/sec",
            "range": "stddev: 0.00012619222627763878",
            "extra": "mean: 3.8052928137650963 msec\nrounds: 247"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 274.1329375978502,
            "unit": "iter/sec",
            "range": "stddev: 0.00014473894668437648",
            "extra": "mean: 3.647865188191973 msec\nrounds: 271"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 414.58571964420486,
            "unit": "iter/sec",
            "range": "stddev: 0.00003444272826616748",
            "extra": "mean: 2.412046417947522 msec\nrounds: 390"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 230.98549146869138,
            "unit": "iter/sec",
            "range": "stddev: 0.0000742036030882312",
            "extra": "mean: 4.329276239999444 msec\nrounds: 50"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 235.11411428059742,
            "unit": "iter/sec",
            "range": "stddev: 0.00017093006483143104",
            "extra": "mean: 4.25325380001027 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 413.51411844243654,
            "unit": "iter/sec",
            "range": "stddev: 0.000048135035474656",
            "extra": "mean: 2.418297115867896 msec\nrounds: 397"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 230.2225522139688,
            "unit": "iter/sec",
            "range": "stddev: 0.00007990931952455347",
            "extra": "mean: 4.3436231176457465 msec\nrounds: 221"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 239.33637031228753,
            "unit": "iter/sec",
            "range": "stddev: 0.000033565183569274833",
            "extra": "mean: 4.178219961701576 msec\nrounds: 235"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_to_matrix",
            "value": 688.1665720107984,
            "unit": "iter/sec",
            "range": "stddev: 0.00003874976758175191",
            "extra": "mean: 1.4531365525036117 msec\nrounds: 619"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_from_matrix",
            "value": 10862.741659345713,
            "unit": "iter/sec",
            "range": "stddev: 0.000007300302292518362",
            "extra": "mean: 92.05779087452147 usec\nrounds: 2893"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body",
            "value": 38653.8164186091,
            "unit": "iter/sec",
            "range": "stddev: 0.000011305659606941323",
            "extra": "mean: 25.8706666676921 usec\nrounds: 9"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body_vmap",
            "value": 30386.032838207597,
            "unit": "iter/sec",
            "range": "stddev: 0.00001137170434258398",
            "extra": "mean: 32.909857148004974 usec\nrounds: 7"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body",
            "value": 341.5392307893317,
            "unit": "iter/sec",
            "range": "stddev: 0.00009680434240733383",
            "extra": "mean: 2.9279213333381904 msec\nrounds: 6"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body_matrix",
            "value": 3.953973955956504,
            "unit": "iter/sec",
            "range": "stddev: 0.0002943663780048697",
            "extra": "mean: 252.91011299999582 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=1,]",
            "value": 564.3379407126966,
            "unit": "iter/sec",
            "range": "stddev: 0.00002388759572862133",
            "extra": "mean: 1.771987895651868 msec\nrounds: 115"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.703629471580945,
            "unit": "iter/sec",
            "range": "stddev: 0.0009298502093580083",
            "extra": "mean: 212.60178039999573 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=1,]",
            "value": 1158.1824338616088,
            "unit": "iter/sec",
            "range": "stddev: 0.00001398536570869583",
            "extra": "mean: 863.4218330058785 usec\nrounds: 1018"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=10000,]",
            "value": 16.878757047805976,
            "unit": "iter/sec",
            "range": "stddev: 0.0038581756086317285",
            "extra": "mean: 59.2460687222219 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 566.2936495618734,
            "unit": "iter/sec",
            "range": "stddev: 0.000019639924982958825",
            "extra": "mean: 1.7658682924904312 msec\nrounds: 506"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.667304400294604,
            "unit": "iter/sec",
            "range": "stddev: 0.0009279081322039043",
            "extra": "mean: 214.2564345999972 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1149.388829016501,
            "unit": "iter/sec",
            "range": "stddev: 0.00004596087393455633",
            "extra": "mean: 870.0275961927274 usec\nrounds: 998"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 17.102942133620665,
            "unit": "iter/sec",
            "range": "stddev: 0.00045072179136301284",
            "extra": "mean: 58.469472222221775 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=1,]",
            "value": 562.589435450266,
            "unit": "iter/sec",
            "range": "stddev: 0.00008781091170747343",
            "extra": "mean: 1.7774951625240782 msec\nrounds: 523"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.708634823650406,
            "unit": "iter/sec",
            "range": "stddev: 0.000983395986460109",
            "extra": "mean: 212.3757813999987 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=1,]",
            "value": 1157.4788113532852,
            "unit": "iter/sec",
            "range": "stddev: 0.000013939510599541385",
            "extra": "mean: 863.9467005282229 usec\nrounds: 945"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=10000,]",
            "value": 17.582926529778934,
            "unit": "iter/sec",
            "range": "stddev: 0.0002509077447279462",
            "extra": "mean: 56.87335372222435 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 565.3232223123813,
            "unit": "iter/sec",
            "range": "stddev: 0.00003079307872827815",
            "extra": "mean: 1.7688995614042347 msec\nrounds: 513"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.685362489314401,
            "unit": "iter/sec",
            "range": "stddev: 0.0006865500650162111",
            "extra": "mean: 213.43065820000788 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1150.147547484638,
            "unit": "iter/sec",
            "range": "stddev: 0.000015296733038627317",
            "extra": "mean: 869.4536646076328 usec\nrounds: 972"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 17.415314227388546,
            "unit": "iter/sec",
            "range": "stddev: 0.00026780550987037425",
            "extra": "mean: 57.4207267777764 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/orbits/tests/test_benchmarks.py::test_benchmark_iterate_real_orbits",
            "value": 19757.75375041692,
            "unit": "iter/sec",
            "range": "stddev: 0.000002264524716606065",
            "extra": "mean: 50.61304096772127 usec\nrounds: 16037"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1406.6425285360945,
            "unit": "iter/sec",
            "range": "stddev: 0.000025324120893849158",
            "extra": "mean: 710.9126730589534 usec\nrounds: 1043"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 17.280398002126926,
            "unit": "iter/sec",
            "range": "stddev: 0.00017320942273699045",
            "extra": "mean: 57.86903749999953 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1431.6668694820175,
            "unit": "iter/sec",
            "range": "stddev: 0.000016142860849352443",
            "extra": "mean: 698.4865133896712 usec\nrounds: 1307"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 19.178630758420823,
            "unit": "iter/sec",
            "range": "stddev: 0.0012039957100506411",
            "extra": "mean: 52.14136569999539 msec\nrounds: 20"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1425.8596771757561,
            "unit": "iter/sec",
            "range": "stddev: 0.00001569028400428867",
            "extra": "mean: 701.3312852641507 usec\nrounds: 1269"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 18.07645280446592,
            "unit": "iter/sec",
            "range": "stddev: 0.0012688238779564058",
            "extra": "mean: 55.32058810525829 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1428.1606716748615,
            "unit": "iter/sec",
            "range": "stddev: 0.000019231456750067204",
            "extra": "mean: 700.2013287673435 usec\nrounds: 1241"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 17.085730920163297,
            "unit": "iter/sec",
            "range": "stddev: 0.0012418328113130723",
            "extra": "mean: 58.52837111111676 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1457.221605986818,
            "unit": "iter/sec",
            "range": "stddev: 0.000014348448037698783",
            "extra": "mean: 686.237423252319 usec\nrounds: 1316"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 19.24302403766193,
            "unit": "iter/sec",
            "range": "stddev: 0.00019681510525031273",
            "extra": "mean: 51.96688410526468 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1445.105467040853,
            "unit": "iter/sec",
            "range": "stddev: 0.000022992115838655504",
            "extra": "mean: 691.9910157475933 usec\nrounds: 1270"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 17.423147295837826,
            "unit": "iter/sec",
            "range": "stddev: 0.004820746314639024",
            "extra": "mean: 57.394911666670446 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1447.7659735384257,
            "unit": "iter/sec",
            "range": "stddev: 0.00003088637003735658",
            "extra": "mean: 690.7193692057432 usec\nrounds: 1273"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 17.591062126749097,
            "unit": "iter/sec",
            "range": "stddev: 0.00018724778709181052",
            "extra": "mean: 56.84705066667877 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1449.0296045865953,
            "unit": "iter/sec",
            "range": "stddev: 0.000014577246846603416",
            "extra": "mean: 690.1170251006001 usec\nrounds: 1235"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 18.126024228620466,
            "unit": "iter/sec",
            "range": "stddev: 0.00044996286652565157",
            "extra": "mean: 55.16929622222556 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1458.8982198727783,
            "unit": "iter/sec",
            "range": "stddev: 0.000015798094311077377",
            "extra": "mean: 685.4487766029381 usec\nrounds: 1325"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 19.23791128683761,
            "unit": "iter/sec",
            "range": "stddev: 0.0005644417265921191",
            "extra": "mean: 51.98069504999694 msec\nrounds: 20"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1427.088859140652,
            "unit": "iter/sec",
            "range": "stddev: 0.000015261971777706917",
            "extra": "mean: 700.7272137224647 usec\nrounds: 1268"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 17.42894043548525,
            "unit": "iter/sec",
            "range": "stddev: 0.00027655253979779495",
            "extra": "mean: 57.375834388876804 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1445.330842651853,
            "unit": "iter/sec",
            "range": "stddev: 0.00003164377170946501",
            "extra": "mean: 691.8831111119359 usec\nrounds: 1296"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 17.945988402633194,
            "unit": "iter/sec",
            "range": "stddev: 0.00022641396568431828",
            "extra": "mean: 55.72275973683741 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1466.433381758943,
            "unit": "iter/sec",
            "range": "stddev: 0.00001654946323576552",
            "extra": "mean: 681.9266476330005 usec\nrounds: 1331"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 18.59925101299654,
            "unit": "iter/sec",
            "range": "stddev: 0.005020045727787543",
            "extra": "mean: 53.76560590000281 msec\nrounds: 20"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "moeyensj@users.noreply.github.com",
            "name": "Joachim Moeyens",
            "username": "moeyensj"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "76b8c629c5094eeb89a3282ebccb9b9cce8237b2",
          "message": "Add query_neocc (#141)\n\n* Add query_neocc\n\n* Remove try/except in query_neocc\n\n* Fix linting issues due to updated black (or ruff) versions\n\n* Fix timescale declaration in query_neocc",
          "timestamp": "2025-02-05T15:25:05-05:00",
          "tree_id": "ea6de0b56a4e1ea1c565180796c471f261791b9f",
          "url": "https://github.com/B612-Asteroid-Institute/adam_core/commit/76b8c629c5094eeb89a3282ebccb9b9cce8237b2"
        },
        "date": 1738787367411,
        "tool": "pytest",
        "benches": [
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 419.0022331244719,
            "unit": "iter/sec",
            "range": "stddev: 0.00009217503942324107",
            "extra": "mean: 2.386622125001736 msec\nrounds: 8"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 287.27033885467176,
            "unit": "iter/sec",
            "range": "stddev: 0.00020687509138395845",
            "extra": "mean: 3.4810416000027544 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 307.5597257909767,
            "unit": "iter/sec",
            "range": "stddev: 0.00015246909045890197",
            "extra": "mean: 3.251401000011356 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 432.21651304167506,
            "unit": "iter/sec",
            "range": "stddev: 0.0000507624073884652",
            "extra": "mean: 2.313655239506266 msec\nrounds: 405"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 298.23918435997024,
            "unit": "iter/sec",
            "range": "stddev: 0.00010271752934136685",
            "extra": "mean: 3.3530134618159866 msec\nrounds: 275"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 317.1405478037187,
            "unit": "iter/sec",
            "range": "stddev: 0.00005797097309659851",
            "extra": "mean: 3.153176113635616 msec\nrounds: 308"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 408.156192658743,
            "unit": "iter/sec",
            "range": "stddev: 0.00006717354768840376",
            "extra": "mean: 2.4500424543015424 msec\nrounds: 372"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 254.58374732354625,
            "unit": "iter/sec",
            "range": "stddev: 0.0000623364096441975",
            "extra": "mean: 3.927980519232112 msec\nrounds: 52"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 257.3205906794304,
            "unit": "iter/sec",
            "range": "stddev: 0.0001728225936722056",
            "extra": "mean: 3.886202800015326 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 404.65897678551846,
            "unit": "iter/sec",
            "range": "stddev: 0.0002920198541792534",
            "extra": "mean: 2.4712166475180664 msec\nrounds: 383"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 255.0036964945249,
            "unit": "iter/sec",
            "range": "stddev: 0.00007350690381793477",
            "extra": "mean: 3.921511780992833 msec\nrounds: 242"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 262.2229264609246,
            "unit": "iter/sec",
            "range": "stddev: 0.0000447424651872478",
            "extra": "mean: 3.8135490801526686 msec\nrounds: 262"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 398.07447232123224,
            "unit": "iter/sec",
            "range": "stddev: 0.00004453911180891685",
            "extra": "mean: 2.5120927603542356 msec\nrounds: 338"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 221.709149351374,
            "unit": "iter/sec",
            "range": "stddev: 0.0003077488592284225",
            "extra": "mean: 4.510413769235829 msec\nrounds: 52"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 227.64350680852382,
            "unit": "iter/sec",
            "range": "stddev: 0.0001618429140828842",
            "extra": "mean: 4.392833399992924 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 396.7848946011723,
            "unit": "iter/sec",
            "range": "stddev: 0.00012726936733641886",
            "extra": "mean: 2.520257231579212 msec\nrounds: 380"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 225.22607861165167,
            "unit": "iter/sec",
            "range": "stddev: 0.00006108012957624239",
            "extra": "mean: 4.439983176745087 msec\nrounds: 215"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 230.0251645799478,
            "unit": "iter/sec",
            "range": "stddev: 0.00003532618988076282",
            "extra": "mean: 4.347350438053654 msec\nrounds: 226"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_to_matrix",
            "value": 652.8813525680304,
            "unit": "iter/sec",
            "range": "stddev: 0.00009608798118318576",
            "extra": "mean: 1.5316718666670754 msec\nrounds: 585"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_from_matrix",
            "value": 10545.524142510912,
            "unit": "iter/sec",
            "range": "stddev: 0.000007522980474882904",
            "extra": "mean: 94.82696037542784 usec\nrounds: 2877"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body",
            "value": 37169.65469820613,
            "unit": "iter/sec",
            "range": "stddev: 0.00001286928699437434",
            "extra": "mean: 26.903666663555565 usec\nrounds: 9"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body_vmap",
            "value": 30371.267036158846,
            "unit": "iter/sec",
            "range": "stddev: 0.000021153914791061836",
            "extra": "mean: 32.92585715338906 usec\nrounds: 7"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body",
            "value": 325.0188738454943,
            "unit": "iter/sec",
            "range": "stddev: 0.00010280978674052627",
            "extra": "mean: 3.076744400004827 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body_matrix",
            "value": 3.8650357080717614,
            "unit": "iter/sec",
            "range": "stddev: 0.00044616425944519025",
            "extra": "mean: 258.7298217999887 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=1,]",
            "value": 542.6034682655629,
            "unit": "iter/sec",
            "range": "stddev: 0.00002494708587441547",
            "extra": "mean: 1.8429664727291724 msec\nrounds: 110"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.773599694722404,
            "unit": "iter/sec",
            "range": "stddev: 0.0008296155214115679",
            "extra": "mean: 209.48551700000735 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=1,]",
            "value": 1102.698445387967,
            "unit": "iter/sec",
            "range": "stddev: 0.000015550104337178927",
            "extra": "mean: 906.8662463273591 usec\nrounds: 885"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=10000,]",
            "value": 17.02272394059536,
            "unit": "iter/sec",
            "range": "stddev: 0.0004401170804875873",
            "extra": "mean: 58.74500482353623 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 541.5753110085948,
            "unit": "iter/sec",
            "range": "stddev: 0.00002432692907692091",
            "extra": "mean: 1.8464652647065185 msec\nrounds: 476"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.751562646407449,
            "unit": "iter/sec",
            "range": "stddev: 0.0011072898586116909",
            "extra": "mean: 210.45708000000332 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1098.9232524536117,
            "unit": "iter/sec",
            "range": "stddev: 0.000014427509187602587",
            "extra": "mean: 909.9816550129943 usec\nrounds: 858"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 16.804830679455854,
            "unit": "iter/sec",
            "range": "stddev: 0.0002646427562100005",
            "extra": "mean: 59.506698941186855 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=1,]",
            "value": 536.9523781340896,
            "unit": "iter/sec",
            "range": "stddev: 0.00004653123060599882",
            "extra": "mean: 1.8623625496827143 msec\nrounds: 473"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.829072576119077,
            "unit": "iter/sec",
            "range": "stddev: 0.0007276934917894814",
            "extra": "mean: 207.0790994000049 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=1,]",
            "value": 1111.319491175762,
            "unit": "iter/sec",
            "range": "stddev: 0.000020067489724834635",
            "extra": "mean: 899.8312437965184 usec\nrounds: 927"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=10000,]",
            "value": 17.43653471888628,
            "unit": "iter/sec",
            "range": "stddev: 0.00017735604664516268",
            "extra": "mean: 57.35084500000198 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 540.6072690724137,
            "unit": "iter/sec",
            "range": "stddev: 0.00009020285391380145",
            "extra": "mean: 1.8497716497889918 msec\nrounds: 474"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.79682298815301,
            "unit": "iter/sec",
            "range": "stddev: 0.002647672002485002",
            "extra": "mean: 208.47131580001133 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1109.9536781440997,
            "unit": "iter/sec",
            "range": "stddev: 0.00004705872182617141",
            "extra": "mean: 900.9384983272925 usec\nrounds: 897"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 17.431271238093345,
            "unit": "iter/sec",
            "range": "stddev: 0.0001479286825553016",
            "extra": "mean: 57.36816244443806 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/orbits/tests/test_benchmarks.py::test_benchmark_iterate_real_orbits",
            "value": 19588.099425621982,
            "unit": "iter/sec",
            "range": "stddev: 0.000004177816573165696",
            "extra": "mean: 51.05140515531394 usec\nrounds: 16759"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1361.4823791565382,
            "unit": "iter/sec",
            "range": "stddev: 0.0000165277478677573",
            "extra": "mean: 734.4935309552204 usec\nrounds: 953"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 17.124745262039184,
            "unit": "iter/sec",
            "range": "stddev: 0.00030232132888350374",
            "extra": "mean: 58.39502922222866 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1373.1421865135244,
            "unit": "iter/sec",
            "range": "stddev: 0.00002942769395779154",
            "extra": "mean: 728.2567019072142 usec\nrounds: 1258"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 19.09652792915924,
            "unit": "iter/sec",
            "range": "stddev: 0.00039765998225399507",
            "extra": "mean: 52.36554015000081 msec\nrounds: 20"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1366.1959081100447,
            "unit": "iter/sec",
            "range": "stddev: 0.00002746845820895737",
            "extra": "mean: 731.9594459797282 usec\nrounds: 1157"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 18.054799488391854,
            "unit": "iter/sec",
            "range": "stddev: 0.0003586190723989216",
            "extra": "mean: 55.386934684206246 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1361.6917992957453,
            "unit": "iter/sec",
            "range": "stddev: 0.00003061484530170679",
            "extra": "mean: 734.3805701974492 usec\nrounds: 1161"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 17.072022719144115,
            "unit": "iter/sec",
            "range": "stddev: 0.0004258960052632378",
            "extra": "mean: 58.57536722222297 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1365.6171286962413,
            "unit": "iter/sec",
            "range": "stddev: 0.00001667237361601917",
            "extra": "mean: 732.2696669414969 usec\nrounds: 1210"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 19.00286452980419,
            "unit": "iter/sec",
            "range": "stddev: 0.00035442693852350855",
            "extra": "mean: 52.62364515789685 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1355.2091094305924,
            "unit": "iter/sec",
            "range": "stddev: 0.00001919184608283623",
            "extra": "mean: 737.8935051729118 usec\nrounds: 1160"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 17.86387149345848,
            "unit": "iter/sec",
            "range": "stddev: 0.0006527302884352568",
            "extra": "mean: 55.97890694445419 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1371.6874522032936,
            "unit": "iter/sec",
            "range": "stddev: 0.000015995043482367374",
            "extra": "mean: 729.0290498711896 usec\nrounds: 1163"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 17.332836687216695,
            "unit": "iter/sec",
            "range": "stddev: 0.002642427171984486",
            "extra": "mean: 57.693960777783104 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1375.7939971287642,
            "unit": "iter/sec",
            "range": "stddev: 0.00001651516410654467",
            "extra": "mean: 726.8530042193572 usec\nrounds: 1185"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 17.85203365115853,
            "unit": "iter/sec",
            "range": "stddev: 0.0008871795696041733",
            "extra": "mean: 56.01602705555643 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1353.1562339628817,
            "unit": "iter/sec",
            "range": "stddev: 0.00004199873005703878",
            "extra": "mean: 739.0129645793959 usec\nrounds: 1214"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 19.000676253067976,
            "unit": "iter/sec",
            "range": "stddev: 0.0008470047630184524",
            "extra": "mean: 52.62970573684362 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1370.0650218196226,
            "unit": "iter/sec",
            "range": "stddev: 0.00003110786741609284",
            "extra": "mean: 729.8923657447085 usec\nrounds: 1121"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 17.53247374283285,
            "unit": "iter/sec",
            "range": "stddev: 0.00020687310634002592",
            "extra": "mean: 57.03701683333722 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1381.790997286585,
            "unit": "iter/sec",
            "range": "stddev: 0.000022895461690006616",
            "extra": "mean: 723.6984478576676 usec\nrounds: 1237"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 17.89934445053275,
            "unit": "iter/sec",
            "range": "stddev: 0.000576482387281047",
            "extra": "mean: 55.86796783332678 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1384.279355497895,
            "unit": "iter/sec",
            "range": "stddev: 0.000028327498247250815",
            "extra": "mean: 722.3975392165852 usec\nrounds: 1224"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 19.286682398646747,
            "unit": "iter/sec",
            "range": "stddev: 0.00028942436129291023",
            "extra": "mean: 51.84924909999893 msec\nrounds: 20"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "akoumjian@users.noreply.github.com",
            "name": "Alec Koumjian",
            "username": "akoumjian"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c04569c1d1e8de4dd0f37c74ddd2eb0df551a5d1",
          "message": "Generate spice kernel (#143)",
          "timestamp": "2025-02-12T09:40:04-05:00",
          "tree_id": "f89af37a06158d857bd13d89c732df2771a0e23a",
          "url": "https://github.com/B612-Asteroid-Institute/adam_core/commit/c04569c1d1e8de4dd0f37c74ddd2eb0df551a5d1"
        },
        "date": 1739371462317,
        "tool": "pytest",
        "benches": [
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 429.8997882089564,
            "unit": "iter/sec",
            "range": "stddev: 0.00008723996798379697",
            "extra": "mean: 2.3261235000049396 msec\nrounds: 8"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 290.7592240878621,
            "unit": "iter/sec",
            "range": "stddev: 0.00021032317123981695",
            "extra": "mean: 3.439271800016286 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 309.6218903676608,
            "unit": "iter/sec",
            "range": "stddev: 0.00017411433518338222",
            "extra": "mean: 3.229745799990269 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 425.0447392650553,
            "unit": "iter/sec",
            "range": "stddev: 0.0001602436967289225",
            "extra": "mean: 2.352693511109206 msec\nrounds: 405"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 295.65919854259784,
            "unit": "iter/sec",
            "range": "stddev: 0.00024710770848675374",
            "extra": "mean: 3.3822725791361523 msec\nrounds: 278"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 320.2133014377543,
            "unit": "iter/sec",
            "range": "stddev: 0.00005014895261079971",
            "extra": "mean: 3.122918365695649 msec\nrounds: 309"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 417.39050780468517,
            "unit": "iter/sec",
            "range": "stddev: 0.00003076677625011627",
            "extra": "mean: 2.3958379055135164 msec\nrounds: 381"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 256.48172261596596,
            "unit": "iter/sec",
            "range": "stddev: 0.00005673535815013107",
            "extra": "mean: 3.8989133018937006 msec\nrounds: 53"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 260.4430229574628,
            "unit": "iter/sec",
            "range": "stddev: 0.00015323746440398168",
            "extra": "mean: 3.839611400007925 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 411.1708230706094,
            "unit": "iter/sec",
            "range": "stddev: 0.00007720048229390211",
            "extra": "mean: 2.432079184344927 msec\nrounds: 396"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 255.71669581331986,
            "unit": "iter/sec",
            "range": "stddev: 0.0001276005745894701",
            "extra": "mean: 3.910577668069148 msec\nrounds: 238"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 264.11551638218174,
            "unit": "iter/sec",
            "range": "stddev: 0.00005276578101010145",
            "extra": "mean: 3.7862220807692912 msec\nrounds: 260"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 396.75576997794775,
            "unit": "iter/sec",
            "range": "stddev: 0.00002955936323805609",
            "extra": "mean: 2.520442235926604 msec\nrounds: 373"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 227.23849972694026,
            "unit": "iter/sec",
            "range": "stddev: 0.00004072637038328213",
            "extra": "mean: 4.400662745096644 msec\nrounds: 51"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 227.35561906450417,
            "unit": "iter/sec",
            "range": "stddev: 0.00018051145700604697",
            "extra": "mean: 4.3983958000012535 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 397.1334120191322,
            "unit": "iter/sec",
            "range": "stddev: 0.00005530142701691068",
            "extra": "mean: 2.518045497395279 msec\nrounds: 384"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 224.29470040829574,
            "unit": "iter/sec",
            "range": "stddev: 0.00017088959694316525",
            "extra": "mean: 4.458420097218731 msec\nrounds: 216"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 233.2304131210599,
            "unit": "iter/sec",
            "range": "stddev: 0.00005210648241068389",
            "extra": "mean: 4.287605491145543 msec\nrounds: 226"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_to_matrix",
            "value": 676.7853056740768,
            "unit": "iter/sec",
            "range": "stddev: 0.00010422378830166675",
            "extra": "mean: 1.4775734514566654 msec\nrounds: 618"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_from_matrix",
            "value": 10735.87684768289,
            "unit": "iter/sec",
            "range": "stddev: 0.000007316592650077406",
            "extra": "mean: 93.14562882824319 usec\nrounds: 3004"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body",
            "value": 39176.595048963034,
            "unit": "iter/sec",
            "range": "stddev: 0.000012817590246436418",
            "extra": "mean: 25.525444433090644 usec\nrounds: 9"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body_vmap",
            "value": 34197.7214528082,
            "unit": "iter/sec",
            "range": "stddev: 0.00001020187869568517",
            "extra": "mean: 29.24171428730915 usec\nrounds: 7"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body",
            "value": 326.1579490537324,
            "unit": "iter/sec",
            "range": "stddev: 0.00010798536279891738",
            "extra": "mean: 3.0659991666652786 msec\nrounds: 6"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body_matrix",
            "value": 3.9111820953460237,
            "unit": "iter/sec",
            "range": "stddev: 0.002370414895513165",
            "extra": "mean: 255.67717780000976 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=1,]",
            "value": 550.1514038377001,
            "unit": "iter/sec",
            "range": "stddev: 0.000021613345735913424",
            "extra": "mean: 1.8176814473693674 msec\nrounds: 114"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.725830159883607,
            "unit": "iter/sec",
            "range": "stddev: 0.0009652409962403713",
            "extra": "mean: 211.60303399998384 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=1,]",
            "value": 1122.59861334853,
            "unit": "iter/sec",
            "range": "stddev: 0.000035685342164254444",
            "extra": "mean: 890.7903395828735 usec\nrounds: 960"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=10000,]",
            "value": 17.353129663884367,
            "unit": "iter/sec",
            "range": "stddev: 0.0012396070684713683",
            "extra": "mean: 57.62649270587872 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 554.0615427421295,
            "unit": "iter/sec",
            "range": "stddev: 0.0000627713068438209",
            "extra": "mean: 1.8048536540739812 msec\nrounds: 503"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.725757215427361,
            "unit": "iter/sec",
            "range": "stddev: 0.0005758504179147442",
            "extra": "mean: 211.6063001999919 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1132.9927889852468,
            "unit": "iter/sec",
            "range": "stddev: 0.00002213433308230779",
            "extra": "mean: 882.6181505494308 usec\nrounds: 910"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 17.241074503338194,
            "unit": "iter/sec",
            "range": "stddev: 0.00020277345521688223",
            "extra": "mean: 58.001025388897965 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=1,]",
            "value": 557.0409483059454,
            "unit": "iter/sec",
            "range": "stddev: 0.0000229521200025056",
            "extra": "mean: 1.7952001608520287 msec\nrounds: 516"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.754158056046563,
            "unit": "iter/sec",
            "range": "stddev: 0.0007935231316186138",
            "extra": "mean: 210.34218639999835 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=1,]",
            "value": 1130.2867785054736,
            "unit": "iter/sec",
            "range": "stddev: 0.00003369348244875439",
            "extra": "mean: 884.7312195602731 usec\nrounds: 1002"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=10000,]",
            "value": 17.744567780789836,
            "unit": "iter/sec",
            "range": "stddev: 0.0002301415127054906",
            "extra": "mean: 56.35527516666786 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 556.3957661005301,
            "unit": "iter/sec",
            "range": "stddev: 0.00003685498833562372",
            "extra": "mean: 1.7972818287393637 msec\nrounds: 508"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.720538917643891,
            "unit": "iter/sec",
            "range": "stddev: 0.001386016116643871",
            "extra": "mean: 211.84021940001685 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1120.9474951494751,
            "unit": "iter/sec",
            "range": "stddev: 0.000018330243739589735",
            "extra": "mean: 892.102443983474 usec\nrounds: 964"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 17.545194982107922,
            "unit": "iter/sec",
            "range": "stddev: 0.001321572392089721",
            "extra": "mean: 56.995661833326494 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/orbits/tests/test_benchmarks.py::test_benchmark_iterate_real_orbits",
            "value": 19713.276955418343,
            "unit": "iter/sec",
            "range": "stddev: 0.000003031355965249944",
            "extra": "mean: 50.72723333931259 usec\nrounds: 15771"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1380.9218891650655,
            "unit": "iter/sec",
            "range": "stddev: 0.000032788644424150996",
            "extra": "mean: 724.1539205411692 usec\nrounds: 1032"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 17.506849333536145,
            "unit": "iter/sec",
            "range": "stddev: 0.000162768539041991",
            "extra": "mean: 57.12050072221726 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1395.3487239700912,
            "unit": "iter/sec",
            "range": "stddev: 0.000027350133725245145",
            "extra": "mean: 716.6667248275885 usec\nrounds: 1301"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 19.498401327101433,
            "unit": "iter/sec",
            "range": "stddev: 0.00021868226485075184",
            "extra": "mean: 51.28625589473682 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1386.4445239748281,
            "unit": "iter/sec",
            "range": "stddev: 0.000015309979493270545",
            "extra": "mean: 721.2693928301424 usec\nrounds: 1311"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 18.31452911547873,
            "unit": "iter/sec",
            "range": "stddev: 0.0004470386847018174",
            "extra": "mean: 54.60145842105428 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1380.6059730815707,
            "unit": "iter/sec",
            "range": "stddev: 0.00001571182073143579",
            "extra": "mean: 724.3196244964505 usec\nrounds: 1241"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 17.304471529569742,
            "unit": "iter/sec",
            "range": "stddev: 0.0013631269599639164",
            "extra": "mean: 57.78853161110457 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1389.7151567134897,
            "unit": "iter/sec",
            "range": "stddev: 0.000027625050657602534",
            "extra": "mean: 719.5719174315408 usec\nrounds: 1308"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 19.401831955702605,
            "unit": "iter/sec",
            "range": "stddev: 0.0002717882767525971",
            "extra": "mean: 51.541524649999815 msec\nrounds: 20"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1384.803450843386,
            "unit": "iter/sec",
            "range": "stddev: 0.00001749601178512374",
            "extra": "mean: 722.1241392711512 usec\nrounds: 1235"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 18.126792922875392,
            "unit": "iter/sec",
            "range": "stddev: 0.00028570764778027577",
            "extra": "mean: 55.1669566842149 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1386.1481188657046,
            "unit": "iter/sec",
            "range": "stddev: 0.000020391576295355395",
            "extra": "mean: 721.4236244957051 usec\nrounds: 1241"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 17.696444864748607,
            "unit": "iter/sec",
            "range": "stddev: 0.0004677942358243614",
            "extra": "mean: 56.50852516665673 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1390.2493206301704,
            "unit": "iter/sec",
            "range": "stddev: 0.000029749343551382983",
            "extra": "mean: 719.2954423072268 usec\nrounds: 1300"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 18.38215195752665,
            "unit": "iter/sec",
            "range": "stddev: 0.00016521905250906977",
            "extra": "mean: 54.400594789477076 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1392.5223641678097,
            "unit": "iter/sec",
            "range": "stddev: 0.000015225913751424273",
            "extra": "mean: 718.1213212310695 usec\nrounds: 1267"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 19.126028295367107,
            "unit": "iter/sec",
            "range": "stddev: 0.0017260797881562615",
            "extra": "mean: 52.284770500011746 msec\nrounds: 20"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1374.1140915269539,
            "unit": "iter/sec",
            "range": "stddev: 0.000018941865368715312",
            "extra": "mean: 727.7416090601124 usec\nrounds: 1192"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 17.448692230832894,
            "unit": "iter/sec",
            "range": "stddev: 0.0025510713195883664",
            "extra": "mean: 57.31088535294007 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1391.3216424359314,
            "unit": "iter/sec",
            "range": "stddev: 0.000028164709669637922",
            "extra": "mean: 718.7410656885894 usec\nrounds: 1294"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 18.179958848401956,
            "unit": "iter/sec",
            "range": "stddev: 0.0001828149741678595",
            "extra": "mean: 55.00562505882138 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1403.4149524665377,
            "unit": "iter/sec",
            "range": "stddev: 0.000015881396469212704",
            "extra": "mean: 712.5476312208833 usec\nrounds: 1326"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 19.200905969402598,
            "unit": "iter/sec",
            "range": "stddev: 0.0013416744374564898",
            "extra": "mean: 52.08087585000101 msec\nrounds: 20"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "akoumjian@users.noreply.github.com",
            "name": "Alec Koumjian",
            "username": "akoumjian"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c04569c1d1e8de4dd0f37c74ddd2eb0df551a5d1",
          "message": "Generate spice kernel (#143)",
          "timestamp": "2025-02-12T09:40:04-05:00",
          "tree_id": "f89af37a06158d857bd13d89c732df2771a0e23a",
          "url": "https://github.com/B612-Asteroid-Institute/adam_core/commit/c04569c1d1e8de4dd0f37c74ddd2eb0df551a5d1"
        },
        "date": 1739371471223,
        "tool": "pytest",
        "benches": [
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 426.5204895642523,
            "unit": "iter/sec",
            "range": "stddev: 0.00011405056972676392",
            "extra": "mean: 2.344553250001269 msec\nrounds: 8"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 290.8725869284809,
            "unit": "iter/sec",
            "range": "stddev: 0.00021009496689671165",
            "extra": "mean: 3.437931399997751 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 312.67799194709744,
            "unit": "iter/sec",
            "range": "stddev: 0.0001848267937342888",
            "extra": "mean: 3.1981783999981417 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 428.66388353617464,
            "unit": "iter/sec",
            "range": "stddev: 0.00007912105154170712",
            "extra": "mean: 2.3328300759809886 msec\nrounds: 408"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 291.37026859552657,
            "unit": "iter/sec",
            "range": "stddev: 0.0005859132819124103",
            "extra": "mean: 3.4320591624541374 msec\nrounds: 277"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 321.56914935295947,
            "unit": "iter/sec",
            "range": "stddev: 0.000031082270159377553",
            "extra": "mean: 3.1097510504727675 msec\nrounds: 317"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 419.26439437503836,
            "unit": "iter/sec",
            "range": "stddev: 0.00005899101394109958",
            "extra": "mean: 2.3851297973695447 msec\nrounds: 380"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 260.50712009293346,
            "unit": "iter/sec",
            "range": "stddev: 0.00007142508867912938",
            "extra": "mean: 3.8386666730769567 msec\nrounds: 52"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 260.94098136147966,
            "unit": "iter/sec",
            "range": "stddev: 0.00019767067668878088",
            "extra": "mean: 3.832284199984315 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 417.14823752016036,
            "unit": "iter/sec",
            "range": "stddev: 0.00008536356531968472",
            "extra": "mean: 2.397229354113407 msec\nrounds: 401"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 256.03011269146197,
            "unit": "iter/sec",
            "range": "stddev: 0.00017010796794326733",
            "extra": "mean: 3.905790570834474 msec\nrounds: 240"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 262.9547503028201,
            "unit": "iter/sec",
            "range": "stddev: 0.00008672425656777173",
            "extra": "mean: 3.802935671815758 msec\nrounds: 259"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 396.28743756855624,
            "unit": "iter/sec",
            "range": "stddev: 0.00004869510576478698",
            "extra": "mean: 2.523420894024691 msec\nrounds: 368"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 225.27020823543734,
            "unit": "iter/sec",
            "range": "stddev: 0.00007350375244024498",
            "extra": "mean: 4.439113400005681 msec\nrounds: 50"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 228.54631198345464,
            "unit": "iter/sec",
            "range": "stddev: 0.00020191108479092554",
            "extra": "mean: 4.375480800024434 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 382.5210592967938,
            "unit": "iter/sec",
            "range": "stddev: 0.0003744536364126931",
            "extra": "mean: 2.6142351530614976 msec\nrounds: 392"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 223.76456374276583,
            "unit": "iter/sec",
            "range": "stddev: 0.00008031087682523813",
            "extra": "mean: 4.468982859813206 msec\nrounds: 214"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 230.45186657829302,
            "unit": "iter/sec",
            "range": "stddev: 0.00003816485099911791",
            "extra": "mean: 4.339300934498021 msec\nrounds: 229"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_to_matrix",
            "value": 655.3764020278915,
            "unit": "iter/sec",
            "range": "stddev: 0.00005004186209464694",
            "extra": "mean: 1.5258407182586384 msec\nrounds: 575"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_from_matrix",
            "value": 10405.085102782365,
            "unit": "iter/sec",
            "range": "stddev: 0.000018773964125904673",
            "extra": "mean: 96.10685449680712 usec\nrounds: 3347"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body",
            "value": 39992.00163367751,
            "unit": "iter/sec",
            "range": "stddev: 0.000011066513571895958",
            "extra": "mean: 25.004999978743097 usec\nrounds: 9"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body_vmap",
            "value": 29842.856038997495,
            "unit": "iter/sec",
            "range": "stddev: 0.000015381300131499413",
            "extra": "mean: 33.50885715138118 usec\nrounds: 7"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body",
            "value": 329.1344373191303,
            "unit": "iter/sec",
            "range": "stddev: 0.00011192195040243338",
            "extra": "mean: 3.0382721666721104 msec\nrounds: 6"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body_matrix",
            "value": 3.7371551706461448,
            "unit": "iter/sec",
            "range": "stddev: 0.012647721669964676",
            "extra": "mean: 267.583216200012 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=1,]",
            "value": 528.1848363385789,
            "unit": "iter/sec",
            "range": "stddev: 0.0001315252205818478",
            "extra": "mean: 1.8932766168224044 msec\nrounds: 107"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.730254319408313,
            "unit": "iter/sec",
            "range": "stddev: 0.0012260571285690482",
            "extra": "mean: 211.40512380000018 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=1,]",
            "value": 1102.5493736658784,
            "unit": "iter/sec",
            "range": "stddev: 0.00001917913165183868",
            "extra": "mean: 906.9888604399538 usec\nrounds: 910"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=10000,]",
            "value": 16.638884069429654,
            "unit": "iter/sec",
            "range": "stddev: 0.001293015496844324",
            "extra": "mean: 60.100184352944886 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 543.1852471539295,
            "unit": "iter/sec",
            "range": "stddev: 0.00006437833388364691",
            "extra": "mean: 1.840992562370931 msec\nrounds: 489"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.697229124080538,
            "unit": "iter/sec",
            "range": "stddev: 0.0018775004404024255",
            "extra": "mean: 212.8914671999837 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1101.7991900515578,
            "unit": "iter/sec",
            "range": "stddev: 0.00003594279490401249",
            "extra": "mean: 907.6064032622911 usec\nrounds: 920"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 16.606680244223263,
            "unit": "iter/sec",
            "range": "stddev: 0.00023705833595520046",
            "extra": "mean: 60.216731176470766 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=1,]",
            "value": 538.9685075630636,
            "unit": "iter/sec",
            "range": "stddev: 0.00007011862821963913",
            "extra": "mean: 1.8553959757713525 msec\nrounds: 454"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.77732660004723,
            "unit": "iter/sec",
            "range": "stddev: 0.0003343369098511169",
            "extra": "mean: 209.32209239998656 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=1,]",
            "value": 1120.243956698888,
            "unit": "iter/sec",
            "range": "stddev: 0.000013358257053425263",
            "extra": "mean: 892.6627044226864 usec\nrounds: 927"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=10000,]",
            "value": 17.15245676849072,
            "unit": "iter/sec",
            "range": "stddev: 0.0003134339844118667",
            "extra": "mean: 58.300686222221685 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 547.4742443026502,
            "unit": "iter/sec",
            "range": "stddev: 0.000024318969831006173",
            "extra": "mean: 1.8265699444432464 msec\nrounds: 468"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.76276339288993,
            "unit": "iter/sec",
            "range": "stddev: 0.0007880969065433476",
            "extra": "mean: 209.9621411999692 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1111.2815222469121,
            "unit": "iter/sec",
            "range": "stddev: 0.00004378810924557855",
            "extra": "mean: 899.8619881468823 usec\nrounds: 928"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 16.96187533840815,
            "unit": "iter/sec",
            "range": "stddev: 0.0010124727728252343",
            "extra": "mean: 58.9557451666691 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/orbits/tests/test_benchmarks.py::test_benchmark_iterate_real_orbits",
            "value": 19940.894241863487,
            "unit": "iter/sec",
            "range": "stddev: 0.000004424543221782517",
            "extra": "mean: 50.14820237603093 usec\nrounds: 16583"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1165.710558848381,
            "unit": "iter/sec",
            "range": "stddev: 0.00026830354000644235",
            "extra": "mean: 857.8458798450891 usec\nrounds: 774"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 16.949438807069924,
            "unit": "iter/sec",
            "range": "stddev: 0.0003351813127357563",
            "extra": "mean: 58.9990035294196 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1371.404109165754,
            "unit": "iter/sec",
            "range": "stddev: 0.000015005634555545697",
            "extra": "mean: 729.1796730930864 usec\nrounds: 1193"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 18.554927481895152,
            "unit": "iter/sec",
            "range": "stddev: 0.0023628978107504716",
            "extra": "mean: 53.89403978947066 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1347.1648756226202,
            "unit": "iter/sec",
            "range": "stddev: 0.00001709555859120315",
            "extra": "mean: 742.2996383704179 usec\nrounds: 1178"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 17.532069844785898,
            "unit": "iter/sec",
            "range": "stddev: 0.0011701619440011033",
            "extra": "mean: 57.03833083333305 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1336.2968540890724,
            "unit": "iter/sec",
            "range": "stddev: 0.00004984187058129536",
            "extra": "mean: 748.3367164563749 usec\nrounds: 1185"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 16.680334745549867,
            "unit": "iter/sec",
            "range": "stddev: 0.000806718775485036",
            "extra": "mean: 59.95083523529341 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1356.0708210017453,
            "unit": "iter/sec",
            "range": "stddev: 0.000014768838535242683",
            "extra": "mean: 737.4246127213978 usec\nrounds: 1242"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 18.88311439054792,
            "unit": "iter/sec",
            "range": "stddev: 0.0002677297963924076",
            "extra": "mean: 52.95736599999401 msec\nrounds: 16"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1350.2315232454387,
            "unit": "iter/sec",
            "range": "stddev: 0.000021613630599448945",
            "extra": "mean: 740.6137264492119 usec\nrounds: 1104"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 17.0203795774815,
            "unit": "iter/sec",
            "range": "stddev: 0.003823007410719857",
            "extra": "mean: 58.7530962777723 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1349.8410719007502,
            "unit": "iter/sec",
            "range": "stddev: 0.000022082277888978175",
            "extra": "mean: 740.8279543545605 usec\nrounds: 1183"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 16.997085860961143,
            "unit": "iter/sec",
            "range": "stddev: 0.0009157961966675945",
            "extra": "mean: 58.833614666664545 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1291.4554403392838,
            "unit": "iter/sec",
            "range": "stddev: 0.00013445133270305283",
            "extra": "mean: 774.3201730113784 usec\nrounds: 1156"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 17.547756585123313,
            "unit": "iter/sec",
            "range": "stddev: 0.000787289075528638",
            "extra": "mean: 56.98734166667111 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1364.5094071571436,
            "unit": "iter/sec",
            "range": "stddev: 0.00002422706457609361",
            "extra": "mean: 732.8641303275639 usec\nrounds: 1220"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 18.662829722459318,
            "unit": "iter/sec",
            "range": "stddev: 0.0006234654938919115",
            "extra": "mean: 53.582442473692765 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1347.7784272969948,
            "unit": "iter/sec",
            "range": "stddev: 0.000022015626933956503",
            "extra": "mean: 741.9617199286431 usec\nrounds: 1114"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 16.859232021437975,
            "unit": "iter/sec",
            "range": "stddev: 0.0011704334389194553",
            "extra": "mean: 59.31468282353629 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1342.0837377678538,
            "unit": "iter/sec",
            "range": "stddev: 0.00002398363158935865",
            "extra": "mean: 745.1099896815637 usec\nrounds: 1163"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 17.364553724323432,
            "unit": "iter/sec",
            "range": "stddev: 0.00055445269101208",
            "extra": "mean: 57.58858050001297 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1360.5549458031403,
            "unit": "iter/sec",
            "range": "stddev: 0.00001980039115550422",
            "extra": "mean: 734.9942044491974 usec\nrounds: 1169"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 18.79633579956089,
            "unit": "iter/sec",
            "range": "stddev: 0.00028379314700979216",
            "extra": "mean: 53.201858631582944 msec\nrounds: 19"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "moeyensj@users.noreply.github.com",
            "name": "Joachim Moeyens",
            "username": "moeyensj"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e56d4b76055f7832689a759e215591f19542ab8a",
          "message": "Add ITRF93 (#138)\n\n* Add support for ITRF93\n\n* Update test data set\n\n* Add unit tests for ITRF93 transformations",
          "timestamp": "2025-02-12T10:52:46-05:00",
          "tree_id": "3ec3e1c37635dc5a5feab0ff913a1c4d91ff36de",
          "url": "https://github.com/B612-Asteroid-Institute/adam_core/commit/e56d4b76055f7832689a759e215591f19542ab8a"
        },
        "date": 1739375806277,
        "tool": "pytest",
        "benches": [
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 427.6581166608586,
            "unit": "iter/sec",
            "range": "stddev: 0.00010193877464352714",
            "extra": "mean: 2.3383164285714235 msec\nrounds: 7"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 294.2210224802594,
            "unit": "iter/sec",
            "range": "stddev: 0.00019014416087286912",
            "extra": "mean: 3.3988054000019474 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 312.77579005392613,
            "unit": "iter/sec",
            "range": "stddev: 0.0001709442605818453",
            "extra": "mean: 3.1971783999892978 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 432.1324198626724,
            "unit": "iter/sec",
            "range": "stddev: 0.00007816690806349774",
            "extra": "mean: 2.3141054779407444 msec\nrounds: 408"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 302.5959518402262,
            "unit": "iter/sec",
            "range": "stddev: 0.00011004588561876465",
            "extra": "mean: 3.304736874100717 msec\nrounds: 278"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 322.24863270864716,
            "unit": "iter/sec",
            "range": "stddev: 0.00007816246857809529",
            "extra": "mean: 3.103193926982847 msec\nrounds: 315"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 420.24917398699847,
            "unit": "iter/sec",
            "range": "stddev: 0.000033156101147684614",
            "extra": "mean: 2.37954066753487 msec\nrounds: 385"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 255.81974919602686,
            "unit": "iter/sec",
            "range": "stddev: 0.00008826843637289618",
            "extra": "mean: 3.909002346936594 msec\nrounds: 49"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 263.62178583067856,
            "unit": "iter/sec",
            "range": "stddev: 0.00016893155375162118",
            "extra": "mean: 3.7933132000034675 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 410.28263001427155,
            "unit": "iter/sec",
            "range": "stddev: 0.00012749842870087347",
            "extra": "mean: 2.4373442277222788 msec\nrounds: 404"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 257.38892530099594,
            "unit": "iter/sec",
            "range": "stddev: 0.00008559433918905074",
            "extra": "mean: 3.885171045454187 msec\nrounds: 242"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 267.0931566863367,
            "unit": "iter/sec",
            "range": "stddev: 0.00010571539349441175",
            "extra": "mean: 3.7440120608344865 msec\nrounds: 263"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 394.63862832608123,
            "unit": "iter/sec",
            "range": "stddev: 0.00009454752223375628",
            "extra": "mean: 2.5339638044092383 msec\nrounds: 363"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 224.9282519039528,
            "unit": "iter/sec",
            "range": "stddev: 0.0002297576567835894",
            "extra": "mean: 4.445862142862394 msec\nrounds: 49"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 226.69231251949685,
            "unit": "iter/sec",
            "range": "stddev: 0.0001805614353449725",
            "extra": "mean: 4.4112655999924755 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 400.23286176539506,
            "unit": "iter/sec",
            "range": "stddev: 0.0000498746886803287",
            "extra": "mean: 2.498545460732735 msec\nrounds: 382"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 225.0132752050353,
            "unit": "iter/sec",
            "range": "stddev: 0.00014123470620310794",
            "extra": "mean: 4.4441822336428185 msec\nrounds: 214"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 233.28537259930468,
            "unit": "iter/sec",
            "range": "stddev: 0.00007768640653717941",
            "extra": "mean: 4.28659537826068 msec\nrounds: 230"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_to_matrix",
            "value": 670.5167986604973,
            "unit": "iter/sec",
            "range": "stddev: 0.0001252407795261006",
            "extra": "mean: 1.4913869451111692 msec\nrounds: 583"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_from_matrix",
            "value": 10499.684423375604,
            "unit": "iter/sec",
            "range": "stddev: 0.000007683815200748896",
            "extra": "mean: 95.24095769713661 usec\nrounds: 2813"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body",
            "value": 43037.49040813354,
            "unit": "iter/sec",
            "range": "stddev: 0.00000857114659430734",
            "extra": "mean: 23.235555570661543 usec\nrounds: 9"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body_vmap",
            "value": 33159.06404049544,
            "unit": "iter/sec",
            "range": "stddev: 0.000010885788300730466",
            "extra": "mean: 30.15766665726005 usec\nrounds: 6"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body",
            "value": 330.0625055861293,
            "unit": "iter/sec",
            "range": "stddev: 0.00010313131633020114",
            "extra": "mean: 3.0297291666746182 msec\nrounds: 6"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body_matrix",
            "value": 3.859414309811336,
            "unit": "iter/sec",
            "range": "stddev: 0.0007669413340387113",
            "extra": "mean: 259.10667259998945 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=1,]",
            "value": 552.1257061607384,
            "unit": "iter/sec",
            "range": "stddev: 0.00004136684969032294",
            "extra": "mean: 1.8111817451022167 msec\nrounds: 102"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.578709290623758,
            "unit": "iter/sec",
            "range": "stddev: 0.0016259157699063825",
            "extra": "mean: 218.4021601999916 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=1,]",
            "value": 1132.5326656291868,
            "unit": "iter/sec",
            "range": "stddev: 0.000039257405183281926",
            "extra": "mean: 882.9767390810247 usec\nrounds: 893"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=10000,]",
            "value": 16.865863460408217,
            "unit": "iter/sec",
            "range": "stddev: 0.001398789856650747",
            "extra": "mean: 59.2913610588305 msec\nrounds: 17"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 554.5988133510899,
            "unit": "iter/sec",
            "range": "stddev: 0.00004271417372950703",
            "extra": "mean: 1.803105192305826 msec\nrounds: 468"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.620295225794455,
            "unit": "iter/sec",
            "range": "stddev: 0.0006996247136677385",
            "extra": "mean: 216.43638580001152 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1146.9439386963263,
            "unit": "iter/sec",
            "range": "stddev: 0.000022009013349218475",
            "extra": "mean: 871.8821960353615 usec\nrounds: 908"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 16.996705980779296,
            "unit": "iter/sec",
            "range": "stddev: 0.0002913914135685765",
            "extra": "mean: 58.83492961111693 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=1,]",
            "value": 560.0307998751256,
            "unit": "iter/sec",
            "range": "stddev: 0.000047880696589221474",
            "extra": "mean: 1.7856160772282128 msec\nrounds: 505"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.648273748034027,
            "unit": "iter/sec",
            "range": "stddev: 0.0023788978721005936",
            "extra": "mean: 215.13362900000175 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=1,]",
            "value": 1143.806706269816,
            "unit": "iter/sec",
            "range": "stddev: 0.0000303451989296917",
            "extra": "mean: 874.273594059613 usec\nrounds: 909"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=10000,]",
            "value": 17.314561243868773,
            "unit": "iter/sec",
            "range": "stddev: 0.0006958222543439126",
            "extra": "mean: 57.75485649999408 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 554.7244218093473,
            "unit": "iter/sec",
            "range": "stddev: 0.0000923312699793806",
            "extra": "mean: 1.8026969080219966 msec\nrounds: 511"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.573939397562271,
            "unit": "iter/sec",
            "range": "stddev: 0.0042548940916829064",
            "extra": "mean: 218.62991900001134 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1131.6527295631847,
            "unit": "iter/sec",
            "range": "stddev: 0.00003650642584454501",
            "extra": "mean: 883.6633128486312 usec\nrounds: 895"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 17.144322117016706,
            "unit": "iter/sec",
            "range": "stddev: 0.0006420367330481705",
            "extra": "mean: 58.328348777782445 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/orbits/tests/test_benchmarks.py::test_benchmark_iterate_real_orbits",
            "value": 19475.86154065491,
            "unit": "iter/sec",
            "range": "stddev: 0.000006132041077804646",
            "extra": "mean: 51.34561045797891 usec\nrounds: 8166"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1386.0997420324263,
            "unit": "iter/sec",
            "range": "stddev: 0.00006356894726292246",
            "extra": "mean: 721.4488031963042 usec\nrounds: 1001"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 17.18693250748063,
            "unit": "iter/sec",
            "range": "stddev: 0.0011684642972646271",
            "extra": "mean: 58.18373927777682 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1423.748965011234,
            "unit": "iter/sec",
            "range": "stddev: 0.000017710565985900188",
            "extra": "mean: 702.3710110244817 usec\nrounds: 1270"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 18.962979018846923,
            "unit": "iter/sec",
            "range": "stddev: 0.002896371381798551",
            "extra": "mean: 52.73433035000039 msec\nrounds: 20"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1400.6534911072852,
            "unit": "iter/sec",
            "range": "stddev: 0.00002136108031999323",
            "extra": "mean: 713.9524560135505 usec\nrounds: 1239"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 18.13576732079364,
            "unit": "iter/sec",
            "range": "stddev: 0.0002785925371113825",
            "extra": "mean: 55.13965757894599 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1404.0631413728127,
            "unit": "iter/sec",
            "range": "stddev: 0.000032817067827463706",
            "extra": "mean: 712.2186820047546 usec\nrounds: 1217"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 17.073737223501094,
            "unit": "iter/sec",
            "range": "stddev: 0.0006027431611252042",
            "extra": "mean: 58.56948522222498 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1401.3279245107196,
            "unit": "iter/sec",
            "range": "stddev: 0.00003896590209472525",
            "extra": "mean: 713.6088438037476 usec\nrounds: 1146"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 18.995953148683174,
            "unit": "iter/sec",
            "range": "stddev: 0.0005846329056113922",
            "extra": "mean: 52.642791449994775 msec\nrounds: 20"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1400.9807971602393,
            "unit": "iter/sec",
            "range": "stddev: 0.000028872431932484348",
            "extra": "mean: 713.7856578955118 usec\nrounds: 1216"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 17.810237124030596,
            "unit": "iter/sec",
            "range": "stddev: 0.00043802406163879473",
            "extra": "mean: 56.14748377778432 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1400.782226818099,
            "unit": "iter/sec",
            "range": "stddev: 0.000022551780902810187",
            "extra": "mean: 713.8868418337355 usec\nrounds: 1157"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 17.45900972968376,
            "unit": "iter/sec",
            "range": "stddev: 0.0006155415701146184",
            "extra": "mean: 57.2770171666611 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1404.2215188535467,
            "unit": "iter/sec",
            "range": "stddev: 0.000016230531345353247",
            "extra": "mean: 712.1383532253752 usec\nrounds: 1240"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 18.041660429197435,
            "unit": "iter/sec",
            "range": "stddev: 0.00027159738311318126",
            "extra": "mean: 55.42727089473792 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1404.7623392975263,
            "unit": "iter/sec",
            "range": "stddev: 0.000030827004012282214",
            "extra": "mean: 711.8641865783972 usec\nrounds: 1222"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 19.077329299006177,
            "unit": "iter/sec",
            "range": "stddev: 0.0005784349091533733",
            "extra": "mean: 52.41823865000299 msec\nrounds: 20"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1366.508038664251,
            "unit": "iter/sec",
            "range": "stddev: 0.00003068370612508617",
            "extra": "mean: 731.7922556661216 usec\nrounds: 1103"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 17.310545002284954,
            "unit": "iter/sec",
            "range": "stddev: 0.0005905371463540488",
            "extra": "mean: 57.76825627777763 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1394.8367879797008,
            "unit": "iter/sec",
            "range": "stddev: 0.00003216982550811901",
            "extra": "mean: 716.9297573864628 usec\nrounds: 1117"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 17.80341476537309,
            "unit": "iter/sec",
            "range": "stddev: 0.0003860904219817515",
            "extra": "mean: 56.16899977778189 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1394.313409668166,
            "unit": "iter/sec",
            "range": "stddev: 0.00003081233993714822",
            "extra": "mean: 717.1988686804576 usec\nrounds: 1226"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 19.077517787722478,
            "unit": "iter/sec",
            "range": "stddev: 0.0014429694375485364",
            "extra": "mean: 52.417720750000285 msec\nrounds: 20"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "moeyensj@users.noreply.github.com",
            "name": "Joachim Moeyens",
            "username": "moeyensj"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e56d4b76055f7832689a759e215591f19542ab8a",
          "message": "Add ITRF93 (#138)\n\n* Add support for ITRF93\n\n* Update test data set\n\n* Add unit tests for ITRF93 transformations",
          "timestamp": "2025-02-12T10:52:46-05:00",
          "tree_id": "3ec3e1c37635dc5a5feab0ff913a1c4d91ff36de",
          "url": "https://github.com/B612-Asteroid-Institute/adam_core/commit/e56d4b76055f7832689a759e215591f19542ab8a"
        },
        "date": 1739375832024,
        "tool": "pytest",
        "benches": [
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 418.8269411260511,
            "unit": "iter/sec",
            "range": "stddev: 0.0000901492955854715",
            "extra": "mean: 2.3876210000040032 msec\nrounds: 8"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 265.98984663636173,
            "unit": "iter/sec",
            "range": "stddev: 0.00020405010777471184",
            "extra": "mean: 3.759541999988869 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 300.9739758628073,
            "unit": "iter/sec",
            "range": "stddev: 0.00020723241104445792",
            "extra": "mean: 3.322546400011106 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 424.06831422003216,
            "unit": "iter/sec",
            "range": "stddev: 0.000040529503359474664",
            "extra": "mean: 2.3581106309233464 msec\nrounds: 401"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 302.4276063008645,
            "unit": "iter/sec",
            "range": "stddev: 0.00006433401877395227",
            "extra": "mean: 3.306576447274355 msec\nrounds: 275"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 317.5919433350282,
            "unit": "iter/sec",
            "range": "stddev: 0.0000849970973656578",
            "extra": "mean: 3.148694483553377 msec\nrounds: 304"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 405.9628664221818,
            "unit": "iter/sec",
            "range": "stddev: 0.000054560684261913474",
            "extra": "mean: 2.4632794836955565 msec\nrounds: 368"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 251.9106280408112,
            "unit": "iter/sec",
            "range": "stddev: 0.00008400401819403595",
            "extra": "mean: 3.969661811322996 msec\nrounds: 53"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 258.65396682659616,
            "unit": "iter/sec",
            "range": "stddev: 0.0001832105135357905",
            "extra": "mean: 3.8661691999891445 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 405.537882187864,
            "unit": "iter/sec",
            "range": "stddev: 0.00007198645850313028",
            "extra": "mean: 2.465860882354644 msec\nrounds: 391"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 255.1313562869918,
            "unit": "iter/sec",
            "range": "stddev: 0.0001350448923630612",
            "extra": "mean: 3.919549578512495 msec\nrounds: 242"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 263.3773597865572,
            "unit": "iter/sec",
            "range": "stddev: 0.00004451671515614606",
            "extra": "mean: 3.796833565384689 msec\nrounds: 260"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 390.586245283638,
            "unit": "iter/sec",
            "range": "stddev: 0.00008902053756389926",
            "extra": "mean: 2.5602540081098213 msec\nrounds: 370"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 223.19177779705063,
            "unit": "iter/sec",
            "range": "stddev: 0.00014249832895113845",
            "extra": "mean: 4.480451788458376 msec\nrounds: 52"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 221.40668790792478,
            "unit": "iter/sec",
            "range": "stddev: 0.00015140653596722975",
            "extra": "mean: 4.516575399998146 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 394.6926882947012,
            "unit": "iter/sec",
            "range": "stddev: 0.00018230356117623437",
            "extra": "mean: 2.5336167343777602 msec\nrounds: 384"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 225.23221300382767,
            "unit": "iter/sec",
            "range": "stddev: 0.00005362337026801335",
            "extra": "mean: 4.4398622500015374 msec\nrounds: 216"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 229.90202173580568,
            "unit": "iter/sec",
            "range": "stddev: 0.0000501207572061974",
            "extra": "mean: 4.349679017390985 msec\nrounds: 230"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_to_matrix",
            "value": 655.9776988713828,
            "unit": "iter/sec",
            "range": "stddev: 0.00008504235416663672",
            "extra": "mean: 1.5244420682601123 msec\nrounds: 586"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_from_matrix",
            "value": 10412.585712006477,
            "unit": "iter/sec",
            "range": "stddev: 0.000008213366801615376",
            "extra": "mean: 96.03762481848543 usec\nrounds: 3441"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body",
            "value": 42768.957327454795,
            "unit": "iter/sec",
            "range": "stddev: 0.00001107582623156819",
            "extra": "mean: 23.38144445148929 usec\nrounds: 9"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body_vmap",
            "value": 33981.24233926913,
            "unit": "iter/sec",
            "range": "stddev: 0.000011578177058639736",
            "extra": "mean: 29.42800001294797 usec\nrounds: 7"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body",
            "value": 328.0560595326466,
            "unit": "iter/sec",
            "range": "stddev: 0.00008181570627568946",
            "extra": "mean: 3.0482594999909907 msec\nrounds: 6"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body_matrix",
            "value": 3.8465408052001564,
            "unit": "iter/sec",
            "range": "stddev: 0.0005535745067169933",
            "extra": "mean: 259.9738441999875 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=1,]",
            "value": 540.4368350930049,
            "unit": "iter/sec",
            "range": "stddev: 0.00002739584659306186",
            "extra": "mean: 1.8503550000027438 msec\nrounds: 110"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.726742095297871,
            "unit": "iter/sec",
            "range": "stddev: 0.0015246176021925216",
            "extra": "mean: 211.5622092000308 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=1,]",
            "value": 1101.4678399295888,
            "unit": "iter/sec",
            "range": "stddev: 0.00001875564945118316",
            "extra": "mean: 907.8794348311839 usec\nrounds: 890"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=10000,]",
            "value": 17.154730863346142,
            "unit": "iter/sec",
            "range": "stddev: 0.00072398713062212",
            "extra": "mean: 58.29295766666102 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 540.8842190481877,
            "unit": "iter/sec",
            "range": "stddev: 0.000028727457057720238",
            "extra": "mean: 1.8488245076917456 msec\nrounds: 455"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.730765431091351,
            "unit": "iter/sec",
            "range": "stddev: 0.000695453825529148",
            "extra": "mean: 211.38228359999403 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1100.0844195307664,
            "unit": "iter/sec",
            "range": "stddev: 0.000039355156517042035",
            "extra": "mean: 909.0211462376163 usec\nrounds: 930"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 17.097123850873437,
            "unit": "iter/sec",
            "range": "stddev: 0.000605471957718113",
            "extra": "mean: 58.48936983333096 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=1,]",
            "value": 540.3437173879113,
            "unit": "iter/sec",
            "range": "stddev: 0.00003071635361948695",
            "extra": "mean: 1.850673872612278 msec\nrounds: 471"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.75646253977589,
            "unit": "iter/sec",
            "range": "stddev: 0.0010629607855513646",
            "extra": "mean: 210.24027659999547 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=1,]",
            "value": 1105.9254704076588,
            "unit": "iter/sec",
            "range": "stddev: 0.000022632959126122853",
            "extra": "mean: 904.2200643334371 usec\nrounds: 886"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=10000,]",
            "value": 17.435195979715196,
            "unit": "iter/sec",
            "range": "stddev: 0.0006523811128863907",
            "extra": "mean: 57.35524861111053 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 543.6232551487888,
            "unit": "iter/sec",
            "range": "stddev: 0.000027853676481946665",
            "extra": "mean: 1.839509238298317 msec\nrounds: 470"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.746362220210571,
            "unit": "iter/sec",
            "range": "stddev: 0.0012486619117736443",
            "extra": "mean: 210.687670599998 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1112.7094864345688,
            "unit": "iter/sec",
            "range": "stddev: 0.000023608014619614515",
            "extra": "mean: 898.7071757645192 usec\nrounds: 916"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 17.43379172789488,
            "unit": "iter/sec",
            "range": "stddev: 0.0003056959433821421",
            "extra": "mean: 57.35986844445051 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/orbits/tests/test_benchmarks.py::test_benchmark_iterate_real_orbits",
            "value": 19879.673685595488,
            "unit": "iter/sec",
            "range": "stddev: 0.00000246466666494582",
            "extra": "mean: 50.30263654300246 usec\nrounds: 16222"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1343.3965871922755,
            "unit": "iter/sec",
            "range": "stddev: 0.000020023824499302157",
            "extra": "mean: 744.3818225636697 usec\nrounds: 975"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 17.265542039523293,
            "unit": "iter/sec",
            "range": "stddev: 0.00016916324489205656",
            "extra": "mean: 57.91883033332271 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1363.0892983693839,
            "unit": "iter/sec",
            "range": "stddev: 0.00001769051521068602",
            "extra": "mean: 733.6276509516032 usec\nrounds: 1209"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 19.249735254540187,
            "unit": "iter/sec",
            "range": "stddev: 0.0002484272474752806",
            "extra": "mean: 51.948766400002455 msec\nrounds: 20"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1344.745891358265,
            "unit": "iter/sec",
            "range": "stddev: 0.000019870411869885183",
            "extra": "mean: 743.6349175158637 usec\nrounds: 1079"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 18.12857248268762,
            "unit": "iter/sec",
            "range": "stddev: 0.00024905683062164004",
            "extra": "mean: 55.161541315786316 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1342.8662372674207,
            "unit": "iter/sec",
            "range": "stddev: 0.000041886629836323096",
            "extra": "mean: 744.6758077966765 usec\nrounds: 1103"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 17.131580695917638,
            "unit": "iter/sec",
            "range": "stddev: 0.00018178254705125373",
            "extra": "mean: 58.37172983333024 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1359.4483757610221,
            "unit": "iter/sec",
            "range": "stddev: 0.000018689772482359363",
            "extra": "mean: 735.592478412575 usec\nrounds: 1158"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 19.194089105019266,
            "unit": "iter/sec",
            "range": "stddev: 0.00020088925159800338",
            "extra": "mean: 52.09937260000004 msec\nrounds: 20"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1345.6363425118852,
            "unit": "iter/sec",
            "range": "stddev: 0.000017329654454115802",
            "extra": "mean: 743.1428302042664 usec\nrounds: 1172"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 18.01700881303414,
            "unit": "iter/sec",
            "range": "stddev: 0.0002602624255540241",
            "extra": "mean: 55.503108777776966 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1353.093234145985,
            "unit": "iter/sec",
            "range": "stddev: 0.000016468589502955855",
            "extra": "mean: 739.0473729115625 usec\nrounds: 1137"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 17.603253558500032,
            "unit": "iter/sec",
            "range": "stddev: 0.00030493860232366505",
            "extra": "mean: 56.80768027778211 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1357.6269439257394,
            "unit": "iter/sec",
            "range": "stddev: 0.000018105059598596458",
            "extra": "mean: 736.5793706983904 usec\nrounds: 1133"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 18.141830323991176,
            "unit": "iter/sec",
            "range": "stddev: 0.00031884368157176834",
            "extra": "mean: 55.12122989473543 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1372.6609451936085,
            "unit": "iter/sec",
            "range": "stddev: 0.000057766032134724185",
            "extra": "mean: 728.5120214875451 usec\nrounds: 1210"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 19.143886963919243,
            "unit": "iter/sec",
            "range": "stddev: 0.00048526647782441994",
            "extra": "mean: 52.23599584999192 msec\nrounds: 20"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1362.265481614063,
            "unit": "iter/sec",
            "range": "stddev: 0.000020640296294617617",
            "extra": "mean: 734.0713051138628 usec\nrounds: 1134"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 17.5370402018242,
            "unit": "iter/sec",
            "range": "stddev: 0.0003676247023945926",
            "extra": "mean: 57.02216499999698 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1366.155003727144,
            "unit": "iter/sec",
            "range": "stddev: 0.000017752059264728977",
            "extra": "mean: 731.9813617574873 usec\nrounds: 1161"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 17.800612128200473,
            "unit": "iter/sec",
            "range": "stddev: 0.0010858227575742231",
            "extra": "mean: 56.17784336841755 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1377.7625654334379,
            "unit": "iter/sec",
            "range": "stddev: 0.000017640755257821843",
            "extra": "mean: 725.8144654884018 usec\nrounds: 1246"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 19.228189528792868,
            "unit": "iter/sec",
            "range": "stddev: 0.00024347410968588362",
            "extra": "mean: 52.0069764499965 msec\nrounds: 20"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "akoumjian@users.noreply.github.com",
            "name": "Alec Koumjian",
            "username": "akoumjian"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ebccd4104907c69657bf01ed9e380642595d702c",
          "message": "Add our first example notebooks for 2024 YR4 (#144)",
          "timestamp": "2025-02-13T21:38:14-05:00",
          "tree_id": "9e82458991dfe043a24869fb620f0f81a29378d1",
          "url": "https://github.com/B612-Asteroid-Institute/adam_core/commit/ebccd4104907c69657bf01ed9e380642595d702c"
        },
        "date": 1739500939731,
        "tool": "pytest",
        "benches": [
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 422.91673726196115,
            "unit": "iter/sec",
            "range": "stddev: 0.00010399279093964351",
            "extra": "mean: 2.364531625005384 msec\nrounds: 8"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 296.4057776124599,
            "unit": "iter/sec",
            "range": "stddev: 0.00021337566154511266",
            "extra": "mean: 3.3737534000010783 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 319.3407376542656,
            "unit": "iter/sec",
            "range": "stddev: 0.00018005096540297432",
            "extra": "mean: 3.131451399985963 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 435.75510178373133,
            "unit": "iter/sec",
            "range": "stddev: 0.00006609392905343056",
            "extra": "mean: 2.2948669927364564 msec\nrounds: 413"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 306.5304801250204,
            "unit": "iter/sec",
            "range": "stddev: 0.00010014065407270526",
            "extra": "mean: 3.2623183168999828 msec\nrounds: 284"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 327.8153525472175,
            "unit": "iter/sec",
            "range": "stddev: 0.00003058636347999605",
            "extra": "mean: 3.0504977641520403 msec\nrounds: 318"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 419.75341955323285,
            "unit": "iter/sec",
            "range": "stddev: 0.0000931807286641113",
            "extra": "mean: 2.382351050443749 msec\nrounds: 337"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 261.73879998668605,
            "unit": "iter/sec",
            "range": "stddev: 0.000076438654447583",
            "extra": "mean: 3.8206028301912722 msec\nrounds: 53"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 263.80537030710275,
            "unit": "iter/sec",
            "range": "stddev: 0.00013944600269739387",
            "extra": "mean: 3.790673399998923 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 409.9505809650031,
            "unit": "iter/sec",
            "range": "stddev: 0.000244017036271103",
            "extra": "mean: 2.4393184116145172 msec\nrounds: 396"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 257.79890408848513,
            "unit": "iter/sec",
            "range": "stddev: 0.00028709185682427907",
            "extra": "mean: 3.8789924400018663 msec\nrounds: 250"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 270.28538943678575,
            "unit": "iter/sec",
            "range": "stddev: 0.000032969404550894814",
            "extra": "mean: 3.699793030188484 msec\nrounds: 265"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 408.3841571672157,
            "unit": "iter/sec",
            "range": "stddev: 0.00006714128475657968",
            "extra": "mean: 2.4486748137747743 msec\nrounds: 392"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 230.09132488391893,
            "unit": "iter/sec",
            "range": "stddev: 0.00004648599547370416",
            "extra": "mean: 4.346100403848341 msec\nrounds: 52"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 232.29451227409098,
            "unit": "iter/sec",
            "range": "stddev: 0.00017075648373663198",
            "extra": "mean: 4.304880000006506 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 405.7562315558564,
            "unit": "iter/sec",
            "range": "stddev: 0.00009207406950667436",
            "extra": "mean: 2.4645339300533697 msec\nrounds: 386"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 228.18475248487988,
            "unit": "iter/sec",
            "range": "stddev: 0.0001507590594507861",
            "extra": "mean: 4.382413763891883 msec\nrounds: 216"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 235.30445887226534,
            "unit": "iter/sec",
            "range": "stddev: 0.0001461756053513653",
            "extra": "mean: 4.249813219828735 msec\nrounds: 232"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_to_matrix",
            "value": 672.4581745889733,
            "unit": "iter/sec",
            "range": "stddev: 0.00009607891385988457",
            "extra": "mean: 1.4870813350008423 msec\nrounds: 600"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_from_matrix",
            "value": 10769.008923233718,
            "unit": "iter/sec",
            "range": "stddev: 0.0000069111741366138255",
            "extra": "mean: 92.85905575233937 usec\nrounds: 2834"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body",
            "value": 41712.047817434766,
            "unit": "iter/sec",
            "range": "stddev: 0.000013959716470709263",
            "extra": "mean: 23.973888896004308 usec\nrounds: 9"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body_vmap",
            "value": 35696.43750524267,
            "unit": "iter/sec",
            "range": "stddev: 0.000009864172884814994",
            "extra": "mean: 28.013999992383887 usec\nrounds: 7"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body",
            "value": 336.4006343169656,
            "unit": "iter/sec",
            "range": "stddev: 0.00010291663120121967",
            "extra": "mean: 2.9726460000006227 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body_matrix",
            "value": 3.9320798107733737,
            "unit": "iter/sec",
            "range": "stddev: 0.0012301813565987454",
            "extra": "mean: 254.31833740000232 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=1,]",
            "value": 564.2285618532056,
            "unit": "iter/sec",
            "range": "stddev: 0.000025592023725567384",
            "extra": "mean: 1.7723314054068895 msec\nrounds: 111"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.6998470901928835,
            "unit": "iter/sec",
            "range": "stddev: 0.0025342713219260004",
            "extra": "mean: 212.7728798000021 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=1,]",
            "value": 1153.8720747467275,
            "unit": "iter/sec",
            "range": "stddev: 0.000014086260726317145",
            "extra": "mean: 866.6471976276035 usec\nrounds: 1012"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=10000,]",
            "value": 17.421525021222653,
            "unit": "iter/sec",
            "range": "stddev: 0.0005719601094899221",
            "extra": "mean: 57.40025622222018 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 564.2255893080516,
            "unit": "iter/sec",
            "range": "stddev: 0.00005355542554794994",
            "extra": "mean: 1.7723407426918876 msec\nrounds: 513"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.706726312159972,
            "unit": "iter/sec",
            "range": "stddev: 0.00047414216264181683",
            "extra": "mean: 212.46189680000498 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1154.7864868130432,
            "unit": "iter/sec",
            "range": "stddev: 0.000015945390195973863",
            "extra": "mean: 865.9609472568217 usec\nrounds: 948"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 17.2123228649744,
            "unit": "iter/sec",
            "range": "stddev: 0.00039159219349841506",
            "extra": "mean: 58.097910888884975 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=1,]",
            "value": 564.6229782300495,
            "unit": "iter/sec",
            "range": "stddev: 0.00003646914646843062",
            "extra": "mean: 1.771093346457042 msec\nrounds: 508"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.721016300505579,
            "unit": "iter/sec",
            "range": "stddev: 0.0011860478072624227",
            "extra": "mean: 211.8187984000201 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=1,]",
            "value": 1157.850439366165,
            "unit": "iter/sec",
            "range": "stddev: 0.000021760093534096773",
            "extra": "mean: 863.6694049599566 usec\nrounds: 968"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=10000,]",
            "value": 17.75044370734713,
            "unit": "iter/sec",
            "range": "stddev: 0.00016728833259099695",
            "extra": "mean: 56.33661988889256 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 563.5875797412597,
            "unit": "iter/sec",
            "range": "stddev: 0.00003411108806091424",
            "extra": "mean: 1.7743471218068632 msec\nrounds: 509"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.686614000017086,
            "unit": "iter/sec",
            "range": "stddev: 0.002241371173504213",
            "extra": "mean: 213.37366379999594 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1150.4123740344633,
            "unit": "iter/sec",
            "range": "stddev: 0.000017502154140131736",
            "extra": "mean: 869.2535151486842 usec\nrounds: 759"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 17.54205197999224,
            "unit": "iter/sec",
            "range": "stddev: 0.0006443102099469144",
            "extra": "mean: 57.00587372221674 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/orbits/tests/test_benchmarks.py::test_benchmark_iterate_real_orbits",
            "value": 19835.63843929407,
            "unit": "iter/sec",
            "range": "stddev: 0.000004335656348590199",
            "extra": "mean: 50.41430872318264 usec\nrounds: 16118"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1411.1886692029177,
            "unit": "iter/sec",
            "range": "stddev: 0.00002667402332579243",
            "extra": "mean: 708.6224697118851 usec\nrounds: 1007"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 17.356122330788995,
            "unit": "iter/sec",
            "range": "stddev: 0.0014172526627129788",
            "extra": "mean: 57.61655633332592 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1437.6401456491474,
            "unit": "iter/sec",
            "range": "stddev: 0.00001563230385075913",
            "extra": "mean: 695.5843595675768 usec\nrounds: 1296"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 19.440763252218897,
            "unit": "iter/sec",
            "range": "stddev: 0.0009476757484325949",
            "extra": "mean: 51.4383096500012 msec\nrounds: 20"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1420.309603712704,
            "unit": "iter/sec",
            "range": "stddev: 0.000027383986316584093",
            "extra": "mean: 704.0718427770886 usec\nrounds: 1253"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 18.338315948685526,
            "unit": "iter/sec",
            "range": "stddev: 0.00019227433216532816",
            "extra": "mean: 54.53063426315757 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1414.1171547190768,
            "unit": "iter/sec",
            "range": "stddev: 0.00001486366493636853",
            "extra": "mean: 707.1549882998601 usec\nrounds: 1282"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 17.34408687275512,
            "unit": "iter/sec",
            "range": "stddev: 0.00041373786855702113",
            "extra": "mean: 57.65653777777402 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1434.942226248504,
            "unit": "iter/sec",
            "range": "stddev: 0.000014195118973183748",
            "extra": "mean: 696.8921686933613 usec\nrounds: 1316"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 19.454218305868757,
            "unit": "iter/sec",
            "range": "stddev: 0.00016701347155788815",
            "extra": "mean: 51.40273354999465 msec\nrounds: 20"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1413.0615567929594,
            "unit": "iter/sec",
            "range": "stddev: 0.00003931703380387099",
            "extra": "mean: 707.6832535658028 usec\nrounds: 1262"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 18.212729562300602,
            "unit": "iter/sec",
            "range": "stddev: 0.00022100340527098786",
            "extra": "mean: 54.906651777773476 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1414.8792021294175,
            "unit": "iter/sec",
            "range": "stddev: 0.0000309208653193062",
            "extra": "mean: 706.7741178858115 usec\nrounds: 1230"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 17.7431571542978,
            "unit": "iter/sec",
            "range": "stddev: 0.000940558905611989",
            "extra": "mean: 56.359755555553825 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1412.0447668303455,
            "unit": "iter/sec",
            "range": "stddev: 0.00003996816565198267",
            "extra": "mean: 708.1928445120949 usec\nrounds: 1312"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 18.381705864580745,
            "unit": "iter/sec",
            "range": "stddev: 0.00015528954253167537",
            "extra": "mean: 54.40191500000418 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1431.205098793025,
            "unit": "iter/sec",
            "range": "stddev: 0.00002140420964320734",
            "extra": "mean: 698.7118763364718 usec\nrounds: 1310"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 19.347158555521514,
            "unit": "iter/sec",
            "range": "stddev: 0.0011443734548240978",
            "extra": "mean: 51.687176549995684 msec\nrounds: 20"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1399.906745122773,
            "unit": "iter/sec",
            "range": "stddev: 0.00003047853661105578",
            "extra": "mean: 714.3332964741871 usec\nrounds: 1248"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 17.656469600653445,
            "unit": "iter/sec",
            "range": "stddev: 0.0004799032504227691",
            "extra": "mean: 56.636463722226274 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1428.5550572693498,
            "unit": "iter/sec",
            "range": "stddev: 0.00001466151530754002",
            "extra": "mean: 700.0080220299503 usec\nrounds: 1271"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 18.170798296878967,
            "unit": "iter/sec",
            "range": "stddev: 0.0004416664731027423",
            "extra": "mean: 55.03335536841884 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1434.5296523414975,
            "unit": "iter/sec",
            "range": "stddev: 0.00001499837915752575",
            "extra": "mean: 697.0925964254274 usec\nrounds: 1343"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 19.35889602829038,
            "unit": "iter/sec",
            "range": "stddev: 0.0007492935028792886",
            "extra": "mean: 51.655838149997635 msec\nrounds: 20"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "akoumjian@users.noreply.github.com",
            "name": "Alec Koumjian",
            "username": "akoumjian"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ebccd4104907c69657bf01ed9e380642595d702c",
          "message": "Add our first example notebooks for 2024 YR4 (#144)",
          "timestamp": "2025-02-13T21:38:14-05:00",
          "tree_id": "9e82458991dfe043a24869fb620f0f81a29378d1",
          "url": "https://github.com/B612-Asteroid-Institute/adam_core/commit/ebccd4104907c69657bf01ed9e380642595d702c"
        },
        "date": 1739500955881,
        "tool": "pytest",
        "benches": [
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 426.97221937207104,
            "unit": "iter/sec",
            "range": "stddev: 0.00010663262796958859",
            "extra": "mean: 2.342072750003865 msec\nrounds: 8"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 294.0402279972335,
            "unit": "iter/sec",
            "range": "stddev: 0.00020018095916469453",
            "extra": "mean: 3.4008951999908277 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 308.7526497915392,
            "unit": "iter/sec",
            "range": "stddev: 0.0001863763661387384",
            "extra": "mean: 3.2388386000093305 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 435.06232642824205,
            "unit": "iter/sec",
            "range": "stddev: 0.00005458306441628305",
            "extra": "mean: 2.298521244553077 msec\nrounds: 413"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 302.5907747010907,
            "unit": "iter/sec",
            "range": "stddev: 0.0001329075953668497",
            "extra": "mean: 3.3047934160842596 msec\nrounds: 286"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[1-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 322.1257083262021,
            "unit": "iter/sec",
            "range": "stddev: 0.00003402169942964965",
            "extra": "mean: 3.104378117462595 msec\nrounds: 315"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 412.29627704692047,
            "unit": "iter/sec",
            "range": "stddev: 0.000044760464552191",
            "extra": "mean: 2.4254402857152098 msec\nrounds: 392"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 256.2632136064824,
            "unit": "iter/sec",
            "range": "stddev: 0.00020694460056831101",
            "extra": "mean: 3.902237804352205 msec\nrounds: 46"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 258.1360614532832,
            "unit": "iter/sec",
            "range": "stddev: 0.00022313018408749456",
            "extra": "mean: 3.8739259999942988 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 410.70925920596545,
            "unit": "iter/sec",
            "range": "stddev: 0.00010915629628652041",
            "extra": "mean: 2.4348124070378283 msec\nrounds: 398"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 258.8822409702343,
            "unit": "iter/sec",
            "range": "stddev: 0.00012030102599306428",
            "extra": "mean: 3.862760134693742 msec\nrounds: 245"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[50-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 265.9001823292445,
            "unit": "iter/sec",
            "range": "stddev: 0.00011013684857607683",
            "extra": "mean: 3.760809756654375 msec\nrounds: 263"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=SphericalCoordinates,]",
            "value": 399.74308501314636,
            "unit": "iter/sec",
            "range": "stddev: 0.00002854938658807572",
            "extra": "mean: 2.5016067506636492 msec\nrounds: 377"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=KeplerianCoordinates,]",
            "value": 230.59429395618452,
            "unit": "iter/sec",
            "range": "stddev: 0.00006961988515651664",
            "extra": "mean: 4.336620749991373 msec\nrounds: 52"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=equatorial,-to=CometaryCoordinates,]",
            "value": 227.34984020482582,
            "unit": "iter/sec",
            "range": "stddev: 0.00014261081021897797",
            "extra": "mean: 4.398507600001267 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=SphericalCoordinates,]",
            "value": 399.5978836036975,
            "unit": "iter/sec",
            "range": "stddev: 0.0000899838602249984",
            "extra": "mean: 2.50251575654428 msec\nrounds: 382"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=KeplerianCoordinates,]",
            "value": 228.56807320403615,
            "unit": "iter/sec",
            "range": "stddev: 0.000044595745819748214",
            "extra": "mean: 4.375064224771798 msec\nrounds: 218"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_transform_cartesian_coordinates[100-origin=SUN,-frame=ecliptic,-to=CometaryCoordinates,]",
            "value": 232.7177233342093,
            "unit": "iter/sec",
            "range": "stddev: 0.00005164083982874104",
            "extra": "mean: 4.2970513189658766 msec\nrounds: 232"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_to_matrix",
            "value": 658.3560712556471,
            "unit": "iter/sec",
            "range": "stddev: 0.00009627974410918522",
            "extra": "mean: 1.5189348798633449 msec\nrounds: 591"
          },
          {
            "name": "src/adam_core/coordinates/tests/test_benchmarks.py::test_benchmark_CoordinateCovariances_from_matrix",
            "value": 10636.431582400073,
            "unit": "iter/sec",
            "range": "stddev: 0.000006784935304199212",
            "extra": "mean: 94.0164934313763 usec\nrounds: 3806"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body",
            "value": 44694.95690431409,
            "unit": "iter/sec",
            "range": "stddev: 0.00001038177693913853",
            "extra": "mean: 22.373888896254357 usec\nrounds: 9"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark__propagate_2body_vmap",
            "value": 35087.19166549605,
            "unit": "iter/sec",
            "range": "stddev: 0.000010400771541134792",
            "extra": "mean: 28.50042857614556 usec\nrounds: 7"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body",
            "value": 333.1448658794632,
            "unit": "iter/sec",
            "range": "stddev: 0.00006761172577825656",
            "extra": "mean: 3.0016971666668724 msec\nrounds: 6"
          },
          {
            "name": "src/adam_core/dynamics/tests/test_propagation.py::test_benchmark_propagate_2body_matrix",
            "value": 3.8990699844695453,
            "unit": "iter/sec",
            "range": "stddev: 0.00014224439113268096",
            "extra": "mean: 256.4714159999994 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=1,]",
            "value": 553.1454449788234,
            "unit": "iter/sec",
            "range": "stddev: 0.000029778281185552685",
            "extra": "mean: 1.8078427818171474 msec\nrounds: 110"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.800919087950541,
            "unit": "iter/sec",
            "range": "stddev: 0.0009747508866445638",
            "extra": "mean: 208.2934499999851 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=1,]",
            "value": 1138.1645132904289,
            "unit": "iter/sec",
            "range": "stddev: 0.000011995624047033734",
            "extra": "mean: 878.6076075320642 usec\nrounds: 1009"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=equatorial,-code=500,-times=10000,]",
            "value": 17.40078582586927,
            "unit": "iter/sec",
            "range": "stddev: 0.00022803793017241885",
            "extra": "mean: 57.46866894444085 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 557.0900297147169,
            "unit": "iter/sec",
            "range": "stddev: 0.000024611732929621703",
            "extra": "mean: 1.795041997991052 msec\nrounds: 498"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.816535155942724,
            "unit": "iter/sec",
            "range": "stddev: 0.00016386958800864343",
            "extra": "mean: 207.61812539999482 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1138.1949439730656,
            "unit": "iter/sec",
            "range": "stddev: 0.000012036506161515751",
            "extra": "mean: 878.584117154244 usec\nrounds: 956"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SUN,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 17.184295385481263,
            "unit": "iter/sec",
            "range": "stddev: 0.00020940753857532653",
            "extra": "mean: 58.19266822222365 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=1,]",
            "value": 553.0159838627347,
            "unit": "iter/sec",
            "range": "stddev: 0.00002055328951094461",
            "extra": "mean: 1.8082659980551523 msec\nrounds: 514"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=X05,-times=10000,]",
            "value": 4.836502544164281,
            "unit": "iter/sec",
            "range": "stddev: 0.0008235259371146435",
            "extra": "mean: 206.76097879997997 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=1,]",
            "value": 1141.1125057556976,
            "unit": "iter/sec",
            "range": "stddev: 0.000014183354468816923",
            "extra": "mean: 876.3377799788057 usec\nrounds: 959"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-code=500,-times=10000,]",
            "value": 17.606316013125745,
            "unit": "iter/sec",
            "range": "stddev: 0.000334289909220066",
            "extra": "mean: 56.797799111096644 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=1,]",
            "value": 557.6780965354245,
            "unit": "iter/sec",
            "range": "stddev: 0.000017950219938408292",
            "extra": "mean: 1.7931491414357146 msec\nrounds: 502"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=X05,-times=10000,]",
            "value": 4.829826067714361,
            "unit": "iter/sec",
            "range": "stddev: 0.00037589342672827834",
            "extra": "mean: 207.04679340000212 msec\nrounds: 5"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=1,]",
            "value": 1141.0175325926882,
            "unit": "iter/sec",
            "range": "stddev: 0.000012996478176748744",
            "extra": "mean: 876.4107223906896 usec\nrounds: 987"
          },
          {
            "name": "src/adam_core/observers/tests/test_benchmarks.py::test_benchmark_get_observer_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-code=500,-times=10000,]",
            "value": 17.529905502629415,
            "unit": "iter/sec",
            "range": "stddev: 0.0001987324402893402",
            "extra": "mean: 57.04537311111027 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/orbits/tests/test_benchmarks.py::test_benchmark_iterate_real_orbits",
            "value": 20090.67124297505,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024921583584630797",
            "extra": "mean: 49.7743449139193 usec\nrounds: 16723"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1369.125697869003,
            "unit": "iter/sec",
            "range": "stddev: 0.000013937127241239015",
            "extra": "mean: 730.3931272026123 usec\nrounds: 1022"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 17.428584038565887,
            "unit": "iter/sec",
            "range": "stddev: 0.0002448456314117914",
            "extra": "mean: 57.37700766667015 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1376.4162676192873,
            "unit": "iter/sec",
            "range": "stddev: 0.00003438181448590764",
            "extra": "mean: 726.5243978332556 usec\nrounds: 1292"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 19.329849466887,
            "unit": "iter/sec",
            "range": "stddev: 0.00015375693163186143",
            "extra": "mean: 51.73346029999095 msec\nrounds: 20"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1360.0985965016089,
            "unit": "iter/sec",
            "range": "stddev: 0.000013154224016923526",
            "extra": "mean: 735.2408145792959 usec\nrounds: 1262"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 18.30616551760564,
            "unit": "iter/sec",
            "range": "stddev: 0.00019733161801641867",
            "extra": "mean: 54.626404368422605 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1364.1978204431932,
            "unit": "iter/sec",
            "range": "stddev: 0.000012911519439483489",
            "extra": "mean: 733.0315186071222 usec\nrounds: 1236"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 17.336951914777863,
            "unit": "iter/sec",
            "range": "stddev: 0.00022247093717246423",
            "extra": "mean: 57.680266111115465 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1370.8706299420567,
            "unit": "iter/sec",
            "range": "stddev: 0.000014435980235908985",
            "extra": "mean: 729.463435978833 usec\nrounds: 1273"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 19.236746457830858,
            "unit": "iter/sec",
            "range": "stddev: 0.0006845170396403299",
            "extra": "mean: 51.983842600000685 msec\nrounds: 20"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1372.8623451679389,
            "unit": "iter/sec",
            "range": "stddev: 0.000013236508836638489",
            "extra": "mean: 728.4051482071004 usec\nrounds: 1255"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SUN,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 18.11046893061454,
            "unit": "iter/sec",
            "range": "stddev: 0.00021142944632127628",
            "extra": "mean: 55.2166817894796 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=1,]",
            "value": 1365.0397591553367,
            "unit": "iter/sec",
            "range": "stddev: 0.0000669834601094369",
            "extra": "mean: 732.5793943311826 usec\nrounds: 1235"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=EARTH,-times=10000,]",
            "value": 17.649529307741563,
            "unit": "iter/sec",
            "range": "stddev: 0.000295487388556785",
            "extra": "mean: 56.65873477778089 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=1,]",
            "value": 1369.3113196257361,
            "unit": "iter/sec",
            "range": "stddev: 0.000023864379988393938",
            "extra": "mean: 730.2941162228343 usec\nrounds: 1239"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SUN,-times=10000,]",
            "value": 18.27711022476295,
            "unit": "iter/sec",
            "range": "stddev: 0.0001666715586195873",
            "extra": "mean: 54.713244473688114 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1351.365073270522,
            "unit": "iter/sec",
            "range": "stddev: 0.000089998031624417",
            "extra": "mean: 739.9924859533614 usec\nrounds: 1317"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=equatorial,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 19.28494027568203,
            "unit": "iter/sec",
            "range": "stddev: 0.0009355991931764893",
            "extra": "mean: 51.853932950001536 msec\nrounds: 20"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=1,]",
            "value": 1352.2112099084538,
            "unit": "iter/sec",
            "range": "stddev: 0.00002826815619395241",
            "extra": "mean: 739.5294408687095 usec\nrounds: 1243"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=EARTH,-times=10000,]",
            "value": 17.52997136699034,
            "unit": "iter/sec",
            "range": "stddev: 0.00030928905392879326",
            "extra": "mean: 57.045158777785645 msec\nrounds: 18"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=1,]",
            "value": 1368.7908901294409,
            "unit": "iter/sec",
            "range": "stddev: 0.000014981047411656892",
            "extra": "mean: 730.5717821554424 usec\nrounds: 1244"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SUN,-times=10000,]",
            "value": 18.02595567794426,
            "unit": "iter/sec",
            "range": "stddev: 0.000920886980200493",
            "extra": "mean: 55.47556078946508 msec\nrounds: 19"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=1,]",
            "value": 1373.1244593954548,
            "unit": "iter/sec",
            "range": "stddev: 0.0000703149994874767",
            "extra": "mean: 728.2661037443538 usec\nrounds: 1282"
          },
          {
            "name": "src/adam_core/utils/tests/test_benchmarks.py::test_benchmark_get_perturber_state[origin=SOLAR_SYSTEM_BARYCENTER,-frame=ecliptic,-perturber=SOLAR_SYSTEM_BARYCENTER,-times=10000,]",
            "value": 19.21853013160728,
            "unit": "iter/sec",
            "range": "stddev: 0.0006281948914499964",
            "extra": "mean: 52.033115599999746 msec\nrounds: 20"
          }
        ]
      }
    ]
  }
}