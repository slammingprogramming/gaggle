"""Property-based tests for this project's pure math, using `hypothesis`
(listed as a dev dependency but previously never actually exercised).

These complement, not replace, the hand-written ground-truth tests
elsewhere (`test_telemetry_analysis.py`, `test_signing.py`,
`test_vehicle_appearance.py`) -- those check specific known values;
these check invariants that must hold for *any* valid input, which is
exactly the kind of regression hypothesis is good at catching (e.g. a
future refactor that breaks `sort_keys=True`'s guarantee, or a signature
scheme that silently stops rejecting tampered input).
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from gaggle.core.signing import (
    EventSigner,
    cryptography_available,
    generate_signing_key,
    verify_signature,
)
from gaggle.detection.telemetry_analysis import (
    _haversine_distance_meters,
    _initial_bearing_degrees,
)
from gaggle.enrichment.vehicle_appearance import (
    VEHICLE_FINGERPRINT_DIMENSIONS,
    _cosine_distance,
)
from gaggle.utils.json import canonical_json_bytes

latitudes = st.floats(min_value=-90.0, max_value=90.0, allow_nan=False, allow_infinity=False)
longitudes = st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False)


@given(lat=latitudes, lon=longitudes)
def test_haversine_distance_from_a_point_to_itself_is_zero(lat: float, lon: float) -> None:
    assert _haversine_distance_meters(lat, lon, lat, lon) == pytest.approx(0.0, abs=1e-6)


@given(lat1=latitudes, lon1=longitudes, lat2=latitudes, lon2=longitudes)
def test_haversine_distance_is_symmetric(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> None:
    forward = _haversine_distance_meters(lat1, lon1, lat2, lon2)
    backward = _haversine_distance_meters(lat2, lon2, lat1, lon1)
    assert forward == pytest.approx(backward, rel=1e-9, abs=1e-6)


@given(lat1=latitudes, lon1=longitudes, lat2=latitudes, lon2=longitudes)
def test_initial_bearing_is_always_in_valid_range(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> None:
    bearing = _initial_bearing_degrees(lat1, lon1, lat2, lon2)
    assert 0.0 <= bearing < 360.0


# -- canonical JSON hashing ---------------------------------------------

json_scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-10_000, max_value=10_000),
    st.floats(allow_nan=False, allow_infinity=False, width=32),
    st.text(max_size=20),
)
json_values = st.recursive(
    json_scalars,
    lambda children: st.one_of(
        st.lists(children, max_size=5),
        st.dictionaries(st.text(max_size=10), children, max_size=5),
    ),
    max_leaves=20,
)
json_payloads = st.dictionaries(
    st.text(min_size=1, max_size=10), json_values, min_size=1, max_size=5
)


@given(payload=json_payloads)
def test_canonical_json_bytes_is_independent_of_dict_key_insertion_order(
    payload: dict[str, object],
) -> None:
    reordered = dict(reversed(list(payload.items())))
    assert canonical_json_bytes(payload) == canonical_json_bytes(reordered)


@given(payload=json_payloads)
def test_canonical_json_bytes_round_trips_through_json_loads(payload: dict[str, object]) -> None:
    assert json.loads(canonical_json_bytes(payload)) == payload


# -- Ed25519 signing ------------------------------------------------------

pytestmark_signing = pytest.mark.skipif(
    not cryptography_available(), reason="cryptography not installed"
)
_SIGNER = EventSigner(generate_signing_key()) if cryptography_available() else None


@pytestmark_signing
@given(payload=json_payloads)
def test_sign_then_verify_always_succeeds_for_arbitrary_payloads(
    payload: dict[str, object],
) -> None:
    assert _SIGNER is not None
    signature = _SIGNER.sign_payload(payload)
    assert verify_signature(payload, signature, _SIGNER.public_key_hex) is True


@pytestmark_signing
@given(payload=json_payloads, data=st.data())
def test_verify_fails_after_any_single_key_is_mutated(
    payload: dict[str, object], data: st.DataObject
) -> None:
    assert _SIGNER is not None
    signature = _SIGNER.sign_payload(payload)

    key = data.draw(st.sampled_from(sorted(payload.keys())))
    mutated = dict(payload)
    mutated[key] = {"__hypothesis_mutation__": payload[key]}
    assume(mutated != payload)

    assert verify_signature(mutated, signature, _SIGNER.public_key_hex) is False


# -- vehicle-appearance fingerprint distance -------------------------------

fingerprint_vectors = st.lists(
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    min_size=VEHICLE_FINGERPRINT_DIMENSIONS,
    max_size=VEHICLE_FINGERPRINT_DIMENSIONS,
).map(lambda values: np.array(values, dtype=np.float64))


@given(vector=fingerprint_vectors)
def test_cosine_distance_from_a_fingerprint_to_itself_is_zero(vector: np.ndarray) -> None:
    # the zero vector is a documented degenerate case (distance defined as 1.0)
    assume(float(np.linalg.norm(vector)) > 0.0)
    assert _cosine_distance(vector, vector) == pytest.approx(0.0, abs=1e-6)


@given(a=fingerprint_vectors, b=fingerprint_vectors)
def test_cosine_distance_is_symmetric(a: np.ndarray, b: np.ndarray) -> None:
    assert _cosine_distance(a, b) == pytest.approx(_cosine_distance(b, a), abs=1e-9)
