"""Enrichment package: local face/plate re-identification, optional local
vehicle/object detection (YOLO-ONNX), optional local transcription
(Whisper), and optional cloud LLM transcript analysis.

Everything in this package runs after event assembly, operating on the
clips/regions that already contributed to a detected event -- enrichment
never runs detection from scratch on benign footage. See
docs/local-ai.md for the full design and docs/forensic-considerations.md
for the scope/intent boundaries around face and plate recognition.
"""
