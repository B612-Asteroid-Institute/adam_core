from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pyarrow as pa
import quivr as qv

from ..coordinates.covariances import CoordinateCovariances

ORBITAL_PARAMETER_NAMES = ("x", "y", "z", "vx", "vy", "vz")
NON_GRAVITATIONAL_PARAMETER_NAMES = (
    "A1",
    "A2",
    "A3",
    "DT",
    "R0",
    "ALN",
    "NK",
    "NM",
    "NN",
    "AMRAT",
    "RHO",
)


def parse_parameter_names(value: str | None) -> list[str]:
    if value in (None, ""):
        return []
    return [name.strip() for name in value.split(",") if name.strip()]


class SolvedStateCovariances(qv.Table):
    """
    Full solved-state covariance matrices for orbital plus extra fitted parameters.

    Notes
    -----
    The first six parameters are expected to be orbital coordinates in the same basis
    as the attached `coordinates` column. Additional parameters are identified by
    `parameter_names`.
    """

    dimension = qv.Int64Column(nullable=True)
    parameter_names = qv.LargeStringColumn(nullable=True)
    values = qv.LargeListColumn(pa.float64(), nullable=True)

    @classmethod
    def from_matrix(
        cls,
        covariances: Sequence[np.ndarray | None],
        parameter_names: Sequence[Sequence[str] | str | None],
    ) -> "SolvedStateCovariances":
        if len(covariances) != len(parameter_names):
            raise ValueError(
                "covariances and parameter_names must have the same length."
            )

        dims: list[int | None] = []
        names_out: list[str | None] = []
        flat_values: list[float] = []
        offsets = [0]

        for cov, names in zip(covariances, parameter_names):
            if cov is None:
                dims.append(None)
                names_out.append(None)
                offsets.append(offsets[-1])
                continue

            cov_array = np.asarray(cov, dtype=np.float64)
            if cov_array.ndim != 2 or cov_array.shape[0] != cov_array.shape[1]:
                raise ValueError(
                    "Solved-state covariance matrices must be square 2D arrays."
                )

            if isinstance(names, str):
                names_list = parse_parameter_names(names)
            elif names is None:
                names_list = []
            else:
                names_list = [str(name) for name in names]

            if len(names_list) != cov_array.shape[0]:
                raise ValueError(
                    "parameter_names length must match covariance dimension."
                )

            dims.append(int(cov_array.shape[0]))
            names_out.append(",".join(names_list))
            flat_values.extend(cov_array.reshape(-1).tolist())
            offsets.append(offsets[-1] + cov_array.size)

        return cls.from_kwargs(
            dimension=dims,
            parameter_names=names_out,
            values=pa.LargeListArray.from_arrays(
                pa.array(offsets, type=pa.int64()),
                pa.array(flat_values, type=pa.float64()),
            ),
        )

    def to_matrix(self) -> list[np.ndarray | None]:
        values = self.values.to_pylist()
        dims = self.dimension.to_pylist()
        matrices: list[np.ndarray | None] = []
        for dim, value in zip(dims, values):
            if dim is None or value is None:
                matrices.append(None)
            else:
                matrices.append(np.asarray(value, dtype=np.float64).reshape(dim, dim))
        return matrices

    def parameter_names_list(self) -> list[list[str]]:
        return [
            parse_parameter_names(value) for value in self.parameter_names.to_pylist()
        ]

    def to_orbital_covariances(self) -> CoordinateCovariances:
        """
        Recover the leading orbital 6x6 covariance block as `CoordinateCovariances`.

        Returns
        -------
        CoordinateCovariances
            The leading 6x6 orbital covariance block for each solved-state covariance row.

        Raises
        ------
        ValueError
            If any solved-state covariance is defined but is smaller than 6x6.
        """
        orbital_covariances = []
        for covariance in self.to_matrix():
            if covariance is None:
                orbital_covariances.append(None)
                continue
            covariance = np.asarray(covariance, dtype=np.float64)
            if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
                raise ValueError("Solved-state covariance must be a square matrix.")
            if covariance.shape[0] < 6:
                raise ValueError("Solved-state covariance must be at least 6x6.")
            orbital_covariances.append(covariance[:6, :6])

        if len(orbital_covariances) == 0:
            return CoordinateCovariances.from_matrix(np.empty((0, 6, 6)))

        if all(covariance is None for covariance in orbital_covariances):
            return CoordinateCovariances.nulls(len(orbital_covariances))

        matrices = []
        for covariance in orbital_covariances:
            if covariance is None:
                matrices.append(np.full((6, 6), np.nan))
            else:
                matrices.append(covariance)
        return CoordinateCovariances.from_matrix(np.asarray(matrices, dtype=np.float64))

    @classmethod
    def nulls(cls, length: int) -> "SolvedStateCovariances":
        return cls.from_kwargs(
            dimension=[None] * length,
            parameter_names=[None] * length,
            values=pa.array([None] * length, type=pa.large_list(pa.float64())),
        )

    def is_all_null(self) -> bool:
        return all(dim is None for dim in self.dimension.to_pylist())


def extract_nongrav_parameter_value(
    parameter_name: str,
    parameter_table,
    index: int,
) -> float:
    if parameter_name in ORBITAL_PARAMETER_NAMES:
        raise ValueError(f"{parameter_name} is not a non-gravitational parameter name.")
    if parameter_name not in NON_GRAVITATIONAL_PARAMETER_NAMES:
        raise ValueError(f"Unsupported solved-state parameter name: {parameter_name}")
    scalar = getattr(parameter_table, parameter_name)[index].as_py()
    if scalar is None:
        raise ValueError(
            f"Missing nominal value for solved-state parameter {parameter_name!r} at row {index}."
        )
    return float(scalar)


def build_solved_state(
    coordinate_values: np.ndarray,
    nongrav_parameters,
    index: int,
    parameter_names: Iterable[str],
) -> np.ndarray:
    state = []
    orbit_lookup = {
        "x": float(coordinate_values[0]),
        "y": float(coordinate_values[1]),
        "z": float(coordinate_values[2]),
        "vx": float(coordinate_values[3]),
        "vy": float(coordinate_values[4]),
        "vz": float(coordinate_values[5]),
    }
    for name in parameter_names:
        if name in orbit_lookup:
            state.append(orbit_lookup[name])
        else:
            state.append(
                extract_nongrav_parameter_value(name, nongrav_parameters, index)
            )
    return np.asarray(state, dtype=np.float64)
