import numpy as np
import pytest

pytest.importorskip("shapely")
from shapely.geometry import box  # noqa: E402

from vishwas_ml.subdivision_weights import SHAPE_TO_CODE, build_weights  # noqa: E402


def test_partial_cells_weighted_by_overlap():
    # 1-degree cells centred on 10.5/11.5 N x 70.5/71.5 E; subdivision covers the west half of the grid.
    subs = {"A": box(70.0, 10.0, 71.0, 12.0), "B": box(71.0, 10.0, 71.5, 12.0)}
    w = build_weights([10.5, 11.5], [70.5, 71.5], subs)
    a = w[w["subdivision_code"] == "A"]
    b = w[w["subdivision_code"] == "B"]
    assert len(a) == 2 and np.allclose(a["frac"], 1.0)
    assert len(b) == 2 and np.allclose(b["frac"], 0.5)
    assert np.isclose(a["weight"].iloc[0], np.cos(np.radians(10.5)))  # area x cos(lat)


def test_mapping_covers_the_33_frontend_codes():
    from vishwas_ml.config import subdivision_codes
    assert sorted(SHAPE_TO_CODE.values()) == sorted(subdivision_codes())
