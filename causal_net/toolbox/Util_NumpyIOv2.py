#!/usr/bin/env python3
"""Versioned, pickle-free storage for NumPy arrays and JSON dictionaries.

The payload can contain:
* JSON-compatible dictionaries, recovered as dictionaries.
* Single float, integer, or string values, recovered as one-element arrays.
* Arbitrary non-object NumPy arrays, including native Unicode arrays.
* Object arrays containing only strings; these are converted to Unicode arrays.

Object arrays containing other Python objects are rejected because loading them
would require pickle. Legacy files without the versioned schema, including
files containing object arrays, are not supported by this safe reader.

Every archive contains ``schema.JSON`` with this contract::

    {
      "format": "Util_NumpyIO",
      "version": 2,
      "records": {
        "payload-key": {
          "encoding": "ndarray" | "json",
          "dtype": "NumPy dtype string",
          "shape": [dimension, ...]
        }
      },
      "metadata": null | {
        "encoding": "json",
        "dtype": "NumPy dtype string",
        "shape": [1]
      }
    }
"""

import json
import logging
import os
import tempfile
import time

import numpy as np

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

log = logging.getLogger(__name__)

_FORMAT_NAME = "Util_NumpyIO"
_SCHEMA_VERSION = 2
_SCHEMA_KEY = "schema.JSON"
_META_KEY = "meta.JSON"
_RESERVED_KEYS = frozenset((_SCHEMA_KEY, _META_KEY))
_SCHEMA_FIELDS = frozenset(("format", "version", "records", "metadata"))
_RECORD_SCHEMA_FIELDS = frozenset(("encoding", "dtype", "shape"))
_PAYLOAD_ENCODINGS = frozenset(("ndarray", "json"))


def _encode_json(value, label):
    """Encode a value as strict JSON and return a one-element Unicode array."""
    try:
        text = json.dumps(value, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be JSON-compatible: {exc}") from exc
    if json.loads(text) != value:
        raise TypeError(f"{label} is not losslessly JSON-compatible")
    return np.array([text])


def _decode_json(record, label):
    """Decode one JSON document stored in a one-element string array."""
    if not isinstance(record, np.ndarray) or record.size != 1:
        raise ValueError(f"{label} must be a one-element NumPy array")
    text = record.reshape(-1)[0]
    if isinstance(text, bytes):
        try:
            text = text.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"invalid UTF-8 in {label}: {exc}") from exc
    if not isinstance(text, str):
        raise ValueError(f"{label} must contain JSON text")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {label}: {exc}") from exc


def _safe_array(value, label):
    """Convert a payload value to an array without retaining Python objects."""
    array = value if isinstance(value, np.ndarray) else np.array([value])
    if not array.dtype.hasobject:
        return array

    if all(isinstance(item, (str, np.str_)) for item in array.flat):
        return array.astype(str)

    raise TypeError(
        f"{label} has object dtype with non-string values; "
        "pickle-backed arrays are not supported"
    )


def json_safe_metadata(value):
    """Return strict-JSON metadata, representing non-finite numbers as null."""
    if isinstance(value, dict):
        return {key: json_safe_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_metadata(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe_metadata(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _record_schema(array, encoding):
    """Describe one stored array with a JSON-compatible schema record."""
    assert isinstance(array, np.ndarray)
    assert encoding in _PAYLOAD_ENCODINGS
    assert not array.dtype.hasobject
    return {
        "encoding": encoding,
        "dtype": array.dtype.str,
        "shape": list(array.shape),
    }


def _validate_record_schema(name, record_schema, array, allowed_encodings):
    """Validate one formal schema record against its stored NumPy array."""
    if not isinstance(record_schema, dict):
        raise ValueError(f"schema record {name!r} must be a JSON object")
    if set(record_schema) != _RECORD_SCHEMA_FIELDS:
        raise ValueError(
            f"schema record {name!r} must have fields "
            f"{sorted(_RECORD_SCHEMA_FIELDS)}"
        )

    encoding = record_schema["encoding"]
    if encoding not in allowed_encodings:
        raise ValueError(f"unsupported encoding {encoding!r} for {name!r}")

    dtype = record_schema["dtype"]
    if not isinstance(dtype, str):
        raise ValueError(f"schema dtype for {name!r} must be a string")
    try:
        declared_dtype = np.dtype(dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid schema dtype {dtype!r} for {name!r}") from exc
    if declared_dtype.hasobject:
        raise ValueError(f"schema for {name!r} declares unsafe object dtype")
    if dtype != array.dtype.str:
        raise ValueError(
            f"schema/data dtype mismatch for {name!r}: "
            f"schema={dtype!r}, data={array.dtype.str!r}"
        )

    shape = record_schema["shape"]
    if not isinstance(shape, list) or any(
        type(dimension) is not int or dimension < 0 for dimension in shape
    ):
        raise ValueError(
            f"schema shape for {name!r} must be a list of non-negative integers"
        )
    if shape != list(array.shape):
        raise ValueError(
            f"schema/data shape mismatch for {name!r}: "
            f"schema={shape}, data={list(array.shape)}"
        )


def _validate_schema(schema, stored):
    """Validate the complete formal schema against all stored data records."""
    if not isinstance(schema, dict):
        raise ValueError(f"{_SCHEMA_KEY} must contain a JSON object")
    if set(schema) != _SCHEMA_FIELDS:
        raise ValueError(
            f"{_SCHEMA_KEY} must have fields {sorted(_SCHEMA_FIELDS)}"
        )
    if schema["format"] != _FORMAT_NAME:
        raise ValueError(f"unsupported NPZ format: {schema['format']!r}")
    if type(schema["version"]) is not int or schema["version"] != _SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema['version']!r}")

    records = schema["records"]
    if not isinstance(records, dict):
        raise ValueError("schema records must be a JSON object")
    reserved = _RESERVED_KEYS.intersection(records)
    if reserved:
        raise ValueError(f"schema uses reserved record names: {sorted(reserved)}")

    metadata_schema = schema["metadata"]
    expected_keys = set(records)
    if metadata_schema is not None:
        expected_keys.add(_META_KEY)
    if expected_keys != set(stored):
        raise ValueError(
            "schema/data key mismatch: "
            f"schema={sorted(expected_keys)}, data={sorted(stored)}"
        )

    for name, record_schema in records.items():
        _validate_record_schema(
            name,
            record_schema,
            stored[name],
            _PAYLOAD_ENCODINGS,
        )

    if metadata_schema is not None:
        _validate_record_schema(
            _META_KEY,
            metadata_schema,
            stored[_META_KEY],
            frozenset(("json",)),
        )


#...!...!..................
def write_data_npz(dataD, outF, metaD=None, verb=1):
    if not isinstance(dataD, dict):
        raise TypeError("dataD must be a dictionary")
    try:
        outF = os.fspath(outF)
    except TypeError as exc:
        raise TypeError("outF must be a filesystem path") from exc
    if not outF:
        raise ValueError("outF must not be empty")

    # np.savez_compressed silently appends this suffix. Normalize it here so
    # subsequent size reporting refers to the file that was actually written.
    suffix = b".npz" if isinstance(outF, bytes) else ".npz"
    if not outF.endswith(suffix):
        outF += suffix
    reserved = _RESERVED_KEYS.intersection(dataD)
    if reserved:
        raise ValueError(f"Keys reserved for internal use: {sorted(reserved)}")

    saveD = {}
    encodings = {}

    for item, rec in dataD.items():
        if not isinstance(item, str):
            raise TypeError(f"data key must be a string, got {type(item).__name__}")
        if verb > 1:
            log.info(f"x={item} {type(rec)}")

        if isinstance(rec, dict):
            saveD[item] = _encode_json(rec, f"data[{item!r}]")
            encodings[item] = "json"
        else:
            saveD[item] = _safe_array(rec, f"data[{item!r}]")
            encodings[item] = "ndarray"

        log.debug(
            "npz-write: %s shape=%s dtype=%s encoding=%s",
            item,
            getattr(saveD[item], "shape", None),
            getattr(saveD[item], "dtype", None),
            encodings[item],
        )

    metadata_schema = None
    if metaD is not None:
        if not isinstance(metaD, dict):
            raise TypeError("metaD must be a dictionary or None")
        saveD[_META_KEY] = _encode_json(metaD, "metaD")
        metadata_schema = _record_schema(saveD[_META_KEY], "json")

    schema = {
        "format": _FORMAT_NAME,
        "version": _SCHEMA_VERSION,
        "records": {
            item: _record_schema(saveD[item], encoding)
            for item, encoding in encodings.items()
        },
        "metadata": metadata_schema,
    }
    _validate_schema(schema, saveD)
    saveD[_SCHEMA_KEY] = _encode_json(schema, _SCHEMA_KEY)

    if verb > 1:
        log.info(f"saving data as npz: {outF}")
    start = time.time()

    # Save to NPZ format
    np.savez_compressed(outF, **saveD)

    if verb > 0:
        xx = os.path.getsize(outF) / 1048576
        log.info(
            f"closed  npz: {outF}  size={xx:.2f} MB, "
            f"elaT={(time.time() - start):.1f} sec"
        )


#...!...!..................
def read_data_npz(inpF, verb=1):
    try:
        inpF = os.fspath(inpF)
    except TypeError as exc:
        raise TypeError("inpF must be a filesystem path") from exc
    if not inpF:
        raise ValueError("inpF must not be empty")

    if verb > 0:
        log.info(f"read data from npz: {inpF}")
        start = time.time()

    stored = {}

    # Pickle is intentionally disabled: NPZ input may come from an untrusted source.
    with np.load(inpF, allow_pickle=False) as npzData:
        for item in npzData.files:
            try:
                obj = npzData[item]
            except ValueError as exc:
                if "Object arrays cannot be loaded" in str(exc):
                    raise ValueError(
                        f"unsafe object array in {item!r}; this file requires "
                        "pickle and is not supported"
                    ) from exc
                raise
            if verb > 1:
                log.info(f"read array: {item} {obj.shape} {obj.dtype}")
            stored[item] = obj

    if _SCHEMA_KEY not in stored:
        raise ValueError(f"missing required {_SCHEMA_KEY!r} record")
    schema = _decode_json(stored.pop(_SCHEMA_KEY), _SCHEMA_KEY)
    _validate_schema(schema, stored)
    records = schema["records"]

    inpMD = None
    if _META_KEY in stored:
        inpMD = _decode_json(stored.pop(_META_KEY), _META_KEY)
        if not isinstance(inpMD, dict):
            raise ValueError(f"{_META_KEY} must contain a JSON object")

    objD = {}
    for item, obj in stored.items():
        encoding = records[item]["encoding"]
        if encoding == "ndarray":
            objD[item] = obj
        elif encoding == "json":
            decoded = _decode_json(obj, f"data[{item!r}]")
            if not isinstance(decoded, dict):
                raise ValueError(f"JSON data item {item!r} must be an object")
            objD[item] = decoded
        else:
            raise ValueError(f"unsupported encoding {encoding!r} for {item!r}")

    if verb > 1:
        log.info(
            f" done npz, num rec:{len(objD)}  "
            f"elaT={(time.time() - start):.1f} sec"
        )

    return objD, inpMD


#=================================
#=================================
#   U N I T   T E S T
#=================================
#=================================

if __name__ == "__main__":
    # Setup basic configuration for logging during test
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print(f"testing npzIO schema version {_SCHEMA_VERSION}")
    verb = 1

    var1 = float(15)  # single variable
    one = np.zeros(shape=5, dtype=np.int16)
    one[3] = 3
    two = np.zeros(shape=(2, 3))
    two[1, 2] = 4

    three = np.empty(2, dtype="object")
    three[0] = "record aaaa"
    three[1] = "much longer record bbb"
    text = "This is text1"
    settings = {"enabled": True, "thresholds": [0.1, 0.2]}

    metaD = {"age": 17, "dom": "white", "dates": [11, 22, 33]}

    normalized_meta = json_safe_metadata({
        "finite": np.float32(1.5),
        "nonfinite": [float("nan"), float("inf"), -float("inf")],
        "tuple": (np.int64(2), True),
    })
    assert normalized_meta == {
        "finite": 1.5,
        "nonfinite": [None, None, None],
        "tuple": [2, True],
    }

    outD = {
        "one": one,
        "two": two,
        "var1": var1,
        "atext": text,
        "three": three,
        "settings": settings,
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        outF = os.path.join(tmp_dir, "abcTest.npz")
        write_data_npz(outD, outF, metaD=metaD, verb=verb)
        big, meta2 = read_data_npz(outF, verb=verb)

        np.testing.assert_array_equal(big["one"], one)
        np.testing.assert_array_equal(big["two"], two)
        np.testing.assert_array_equal(big["var1"], np.array([var1]))
        np.testing.assert_array_equal(big["atext"], np.array([text]))
        np.testing.assert_array_equal(big["three"], three.astype(str))
        assert big["three"].dtype.kind == "U"
        assert isinstance(big["three"][1], np.str_)
        assert big["settings"] == settings
        assert meta2 == metaD

        with np.load(outF, allow_pickle=False) as archive:
            raw_records = {key: archive[key] for key in archive.files}

        assert all(array.dtype.kind != "O" for array in raw_records.values())
        stored_schema = _decode_json(
            raw_records.pop(_SCHEMA_KEY),
            _SCHEMA_KEY,
        )
        _validate_schema(stored_schema, raw_records)
        assert set(stored_schema) == _SCHEMA_FIELDS
        assert stored_schema["format"] == _FORMAT_NAME
        assert stored_schema["version"] == _SCHEMA_VERSION
        assert set(stored_schema["records"]) == set(outD)
        for name, record_schema in stored_schema["records"].items():
            expected_encoding = "json" if name == "settings" else "ndarray"
            assert record_schema == _record_schema(
                raw_records[name],
                expected_encoding,
            )
        assert stored_schema["metadata"] == _record_schema(
            raw_records[_META_KEY],
            "json",
        )

        inconsistent_schema = {
            **stored_schema,
            "records": {
                name: dict(record_schema)
                for name, record_schema in stored_schema["records"].items()
            },
        }
        inconsistent_schema["records"]["one"]["shape"] = [999]
        inconsistent_file = os.path.join(tmp_dir, "inconsistent_schema.npz")
        np.savez_compressed(
            inconsistent_file,
            **raw_records,
            **{_SCHEMA_KEY: _encode_json(inconsistent_schema, _SCHEMA_KEY)},
        )
        try:
            read_data_npz(inconsistent_file, verb=0)
        except ValueError as exc:
            assert "schema/data shape mismatch" in str(exc)
        else:
            raise AssertionError("inconsistent schema was accepted")

        try:
            write_data_npz(
                {"unsafe": np.array([object()], dtype=object)},
                os.path.join(tmp_dir, "unsafe.npz"),
                verb=0,
            )
        except TypeError as exc:
            assert "pickle-backed arrays are not supported" in str(exc)
        else:
            raise AssertionError("non-string object array was accepted")

        legacy_file = os.path.join(tmp_dir, "unsafe_legacy.npz")
        np.savez_compressed(
            legacy_file,
            unsafe=np.array([object()], dtype=object),
        )
        try:
            read_data_npz(legacy_file, verb=0)
        except ValueError as exc:
            assert "unsafe object array" in str(exc)
        else:
            raise AssertionError("pickle-backed input file was accepted")

        try:
            write_data_npz(
                {"one": one},
                os.path.join(tmp_dir, "bad_meta.npz"),
                metaD={"unsupported": object()},
                verb=0,
            )
        except TypeError as exc:
            assert "metaD must be JSON-compatible" in str(exc)
        else:
            raise AssertionError("non-JSON metadata was accepted")

    print("all schema, round-trip, and pickle-safety assertions passed")
