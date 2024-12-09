window.BENCHMARK_DATA = {
  "lastUpdate": 1733757149106,
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
      }
    ]
  }
}