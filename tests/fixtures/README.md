# Test fixtures

`sample_face.jpg` -- a standard computer-vision test image (a U.S. Navy
photograph of Grace Hopper, in the public domain as a work of the U.S.
federal government). It is widely bundled as sample/test data by several
open-source projects (e.g. matplotlib's `mpl-data/sample_data/`) for
exactly this purpose: a real photographic face to test detectors against
without needing network access or a live camera. Used by
`tests/unit/test_face_recognition.py` to verify real Haar-cascade face
detection against real content, not just synthetic negative cases.
