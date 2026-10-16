import bz2
import csv
import gzip
import importlib.util
import json
import logging
import typing
import zipfile
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import AsyncIterator

import pandas as pd
import pytest
import yaml
from hamcrest import assert_that, equal_to

from nodestream.pipeline.extractors.files import (
    SUPPORTED_ARCHIVE_FORMAT_REGISTRY,
    SUPPORTED_FILE_FORMAT_REGISTRY,
    FileCodec,
    FileExtractor,
    FileSource,
    LocalFileSource,
    ReadableFile,
    RemoteFileSource,
    ZipArchive,
    _is_data_member,
)

SIMPLE_RECORD = {"record": "value"}


@pytest.fixture(autouse=True)
def base_dir():
    with TemporaryDirectory() as tempdir:
        yield tempdir


@pytest.fixture(autouse=True)
def unsupported_file(base_dir: str):
    with NamedTemporaryFile(
        "w+",
        suffix=".unsupported",
        dir=base_dir,
        delete=False,
    ) as temp_file:
        temp_file.write("hello world")
        temp_file.seek(0)
        yield temp_file


@pytest.fixture(autouse=True)
def json_file(base_dir: str):
    with NamedTemporaryFile(
        "w+",
        suffix=".json",
        dir=base_dir,
        delete=False,
    ) as temp_file:
        json.dump(SIMPLE_RECORD, temp_file)
        temp_file.seek(0)
        yield temp_file


@pytest.fixture
def empty_json_file(base_dir: str):
    with NamedTemporaryFile(
        "w+",
        suffix=".json",
        dir=base_dir,
        delete=False,
    ) as temp_file:
        yield temp_file


@pytest.fixture(autouse=True)
def jsonl_file(base_dir: str):
    with NamedTemporaryFile(
        "w+",
        suffix=".jsonl",
        dir=base_dir,
        delete=False,
    ) as temp_file:
        json.dump(SIMPLE_RECORD, temp_file)
        temp_file.write("\n")
        json.dump(SIMPLE_RECORD, temp_file)
        temp_file.seek(0)
        yield temp_file


@pytest.fixture(autouse=True)
def csv_file(base_dir: str):
    with NamedTemporaryFile(
        "w+",
        suffix=".csv",
        dir=base_dir,
        delete=False,
    ) as temp_file:
        writer = csv.DictWriter(temp_file, SIMPLE_RECORD.keys())
        writer.writeheader()
        writer.writerow(SIMPLE_RECORD)
        temp_file.seek(0)
        yield temp_file


@pytest.fixture(autouse=True)
def txt_file(base_dir: str):
    with NamedTemporaryFile(
        "w+",
        suffix=".txt",
        dir=base_dir,
        delete=False,
    ) as temp_file:
        temp_file.write("hello world")
        temp_file.seek(0)
        yield temp_file


@pytest.fixture(autouse=True)
def yaml_file(base_dir: str):
    with NamedTemporaryFile(
        "w+",
        suffix=".yaml",
        dir=base_dir,
        delete=False,
    ) as temp_file:
        yaml.dump(SIMPLE_RECORD, temp_file)
        temp_file.seek(0)
        yield temp_file


@pytest.fixture(autouse=True)
def parquet_file(base_dir: str):
    with NamedTemporaryFile(
        "wb",
        suffix=".parquet",
        dir=base_dir,
        delete=False,
    ) as temp_file:
        df = pd.DataFrame(data=SIMPLE_RECORD, index=[0])
        temp_file.write(df.to_parquet())
        temp_file.seek(0)
        yield temp_file


@pytest.fixture(autouse=True)
def gzip_file(base_dir: str):
    with NamedTemporaryFile(
        "wb",
        suffix=".json.gz",
        dir=base_dir,
        delete=False,
    ) as temp_file:
        json_data = json.dumps(SIMPLE_RECORD).encode("utf-8")
        temp_file.write(gzip.compress(json_data))
        temp_file.seek(0)
        yield temp_file


@pytest.fixture(autouse=True)
def bz2_file(base_dir: str):
    with NamedTemporaryFile(
        "wb",
        suffix=".json.bz2",
        dir=base_dir,
        delete=False,
    ) as temp_file:
        json_data = json.dumps(SIMPLE_RECORD).encode("utf-8")
        temp_file.write(bz2.compress(json_data))
        temp_file.seek(0)
        yield temp_file


@pytest.fixture(scope="function")
def file_path(
    request,
    unsupported_file,
    json_file,
    empty_json_file,
    csv_file,
    txt_file,
    yaml_file,
    parquet_file,
    gzip_file,
    bz2_file,
    jsonl_file,
):
    # This is a hack to allow us to use the auto-cleanup of the file fixtures
    # but also do a parametrized test
    match request.param:
        case "unsupported":
            return Path(unsupported_file.name)
        case "json":
            return Path(json_file.name)
        case "empty_json":
            return Path(empty_json_file.name)
        case "csv":
            return Path(csv_file.name)
        case "txt":
            return Path(txt_file.name)
        case "yaml":
            return Path(yaml_file.name)
        case "parquet":
            return Path(parquet_file.name)
        case "gzip":
            return Path(gzip_file.name)
        case "bz2":
            return Path(bz2_file.name)
        case "jsonl":
            return Path(jsonl_file.name)
        case _:
            raise ValueError("unknown path type")


@pytest.mark.parametrize(
    "file_path,expected",
    [
        ("unsupported", []),
        ("json", [SIMPLE_RECORD]),
        ("empty_json", []),
        ("csv", [SIMPLE_RECORD]),
        ("txt", [{"line": "hello world"}]),
        ("yaml", [SIMPLE_RECORD]),
        ("parquet", [SIMPLE_RECORD]),
        ("gzip", [SIMPLE_RECORD]),
        ("bz2", [SIMPLE_RECORD]),
        ("jsonl", [SIMPLE_RECORD, SIMPLE_RECORD]),
    ],
    indirect=["file_path"],
)
@pytest.mark.asyncio
async def test_file(file_path, expected):
    subject = FileExtractor([LocalFileSource([file_path])])
    assert [r async for r in subject.extract_records()] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "glob, expected",
    [
        ("*.json", [SIMPLE_RECORD]),
        ("*.csv", [SIMPLE_RECORD]),
        ("*.gz", [SIMPLE_RECORD]),
        ("*.bz2", [SIMPLE_RECORD]),
        ("*.yaml", [SIMPLE_RECORD]),
        ("*.parquet", [SIMPLE_RECORD]),
        ("*.unsupported", []),
        ("*.txt", [{"line": "hello world"}]),
        ("*.jsonl", [SIMPLE_RECORD, SIMPLE_RECORD]),
    ],
)
async def test_formatting_unified(glob: str, expected, base_dir: str):
    subject = FileExtractor.from_file_data(
        [{"type": "local", "globs": [f"{base_dir}/{glob}"]}]
    )
    results = [r async for r in subject.extract_records()]
    assert_that(results, equal_to(expected))


def test_declarative_init(
    base_dir,
    csv_file,
    json_file,
    jsonl_file,
    txt_file,
    yaml_file,
    bz2_file,
    gzip_file,
    unsupported_file,
    parquet_file,
):
    subject = FileExtractor.local(globs=[f"{base_dir}/**"])
    assert isinstance(subject.file_sources[0], LocalFileSource)
    local_source: LocalFileSource = typing.cast(
        LocalFileSource,
        subject.file_sources[0],
    )
    assert set(local_source.paths) == {
        Path(f.name)
        for f in [
            csv_file,
            json_file,
            txt_file,
            jsonl_file,
            yaml_file,
            bz2_file,
            gzip_file,
            unsupported_file,
            parquet_file,
        ]
    }


@pytest.mark.asyncio
async def test_remote_file_extractor_extract_records(httpx_mock):
    files = ["https://example.com/file.json", "https://example.com/file2.json"]
    for file in files:
        httpx_mock.add_response(
            url=file,
            method="GET",
            json=SIMPLE_RECORD,
        )
    subject = FileExtractor.remote(urls=files)
    results = [r async for r in subject.extract_records()]
    assert_that(results, equal_to([SIMPLE_RECORD, SIMPLE_RECORD]))


@pytest.mark.asyncio
async def test_no_files_found_from_local_source(mocker):
    file_source = LocalFileSource([])
    subject = FileExtractor([file_source])
    subject.logger = mocker.Mock()
    results = [r async for r in subject.extract_records()]
    assert_that(results, equal_to([]))
    subject.logger.warning.assert_called_once_with(
        "No files found for source: %s",
        "0 local files",
    )


@pytest.mark.asyncio
async def test_no_files_found_from_remote_source(mocker):
    subject = FileExtractor([RemoteFileSource([], 10)])
    subject.logger = mocker.Mock()
    results = [r async for r in subject.extract_records()]
    assert_that(results, equal_to([]))
    subject.logger.warning.assert_called_once_with(
        "No files found for source: %s",
        "0 remote files",
    )


def test_remote_file_source_single_file_description():
    url = "https://example.com/file.json"
    subject = RemoteFileSource([url], 10)
    assert_that(subject.describe(), equal_to(url))


def test_remote_file_source_multiple_file_description():
    urls = ["https://example.com/file.json", "https://example.com/file2.json"]
    subject = RemoteFileSource(urls, 10)
    assert_that(subject.describe(), equal_to("2 remote files"))


def test_local_file_source_single_file_description(base_dir: str):
    path = Path(f"{base_dir}/file.json")
    subject = LocalFileSource([path])
    assert_that(subject.describe(), equal_to(str(path)))


def test_local_file_source_multiple_file_description(base_dir: str):
    paths = [
        Path(f"{base_dir}/file.json"),
        Path(f"{base_dir}/file2.json"),
    ]
    subject = LocalFileSource(paths)
    assert_that(subject.describe(), equal_to("2 local files"))


def test_file_source_default_description():
    class SomeFileSource(FileSource):
        def get_files(self) -> AsyncIterator[ReadableFile]:
            pass  # not implemented for this test

        def describe(self) -> str:
            return "SomeFileSource(test)"

    subject = SomeFileSource()
    assert_that(subject.describe(), equal_to("SomeFileSource(test)"))


@pytest.mark.asyncio
async def test_unified_extractor_multiple_source_types(json_file, csv_file, httpx_mock):
    urls = ["https://example.com/file.json", "https://example.com/file2.json"]
    for url in urls:
        httpx_mock.add_response(
            url=url,
            method="GET",
            json=SIMPLE_RECORD,
        )

    subject = FileExtractor.from_file_data(
        [
            {"type": "local", "globs": [json_file.name]},
            {"type": "local", "globs": [csv_file.name]},
            {"type": "http", "urls": urls},
        ]
    )
    results = [r async for r in subject.extract_records()]
    assert_that(results, equal_to([SIMPLE_RECORD] * 4))


def _warning_message(logger) -> str:
    logger.warning.assert_called_once()
    args = logger.warning.call_args.args
    return args[0] % args[1:]


async def _records_from(path: Path):
    subject = FileExtractor([LocalFileSource([path])])
    return [record async for record in subject.extract_records()]


@pytest.mark.asyncio
async def test_suffix_lookup_is_case_insensitive(base_dir: str):
    path = Path(base_dir) / "DATA.JSON"
    path.write_text(json.dumps(SIMPLE_RECORD), encoding="utf-8")
    assert await _records_from(path) == [SIMPLE_RECORD]


@pytest.mark.asyncio
async def test_zip_reads_csv_and_skips_directories(base_dir: str):
    archive_path = Path(base_dir) / "notes.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("notes.csv", "record\nvalue\n")
        archive.writestr("folder/", b"")
        archive.writestr("__MACOSX/._notes.csv", b"not a csv")
    assert await _records_from(archive_path) == [SIMPLE_RECORD]


def _corrupt_zip_member_crc(path: Path, member: str) -> None:
    """Flip one stored payload byte so the member's CRC no longer matches."""
    with zipfile.ZipFile(path) as archive:
        offset = archive.getinfo(member).header_offset
    raw = bytearray(path.read_bytes())
    name_len = int.from_bytes(raw[offset + 26 : offset + 28], "little")
    extra_len = int.from_bytes(raw[offset + 28 : offset + 30], "little")
    data_start = offset + 30 + name_len + extra_len
    raw[data_start] ^= 0xFF
    path.write_bytes(raw)


@pytest.mark.asyncio
async def test_zip_bad_member_does_not_drop_later_members(
    base_dir: str, mocker, caplog
):
    archive_path = Path(base_dir) / "bundle.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("a_corrupt.csv", "record\nvalue\n")
        archive.writestr("a_bad.json", b"this is not json")
        archive.writestr("b_good.csv", "record\nvalue\n")
    _corrupt_zip_member_crc(archive_path, "a_corrupt.csv")
    subject = FileExtractor([LocalFileSource([archive_path])])
    subject.logger = mocker.Mock()
    with caplog.at_level(
        logging.WARNING, logger="nodestream.pipeline.extractors.files"
    ):
        records = [record async for record in subject.extract_records()]
    assert records == [SIMPLE_RECORD]
    warning = " ".join(
        str(arg) for call in subject.logger.warning.call_args_list for arg in call.args
    )
    assert "a_bad.json" in warning
    assert "a_corrupt.csv" in caplog.text
    assert "b_good.csv" not in warning
    assert "b_good.csv" not in caplog.text


@pytest.mark.asyncio
async def test_zip_encrypted_member_does_not_drop_later_members(
    base_dir: str, mocker, caplog
):
    archive_path = Path(base_dir) / "bundle.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("first.csv", "record\nfirst\n")
        archive.writestr("secret.csv", "record\nsecret\n")
        archive.writestr("notes.csv", "record\nvalue\n")
        # Traditional encryption is flag bit 0. ZipFile.open raises
        # RuntimeError for that flag when no password is supplied.
        archive.getinfo("secret.csv").flag_bits |= 1
    subject = FileExtractor([LocalFileSource([archive_path])])
    subject.logger = mocker.Mock()
    with caplog.at_level(
        logging.WARNING, logger="nodestream.pipeline.extractors.files"
    ):
        records = [record async for record in subject.extract_records()]
    assert records == [{"record": "first"}, SIMPLE_RECORD]
    subject.logger.warning.assert_not_called()
    assert (
        f"Skipping member secret.csv in {archive_path} that could not be unpacked: "
    ) in caplog.text
    assert "is encrypted, password required for extraction" in caplog.text
    assert "first.csv" not in caplog.text
    assert "notes.csv" not in caplog.text


@pytest.mark.asyncio
async def test_zip_skips_oversized_members(base_dir: str, monkeypatch, caplog):
    limit = 32
    monkeypatch.setattr(ZipArchive, "max_member_bytes", limit)
    archive_path = Path(base_dir) / "bundle.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("big.csv", "x" * (limit + 1))
        archive.writestr("notes.csv", "record\nvalue\n")
    with caplog.at_level(
        logging.WARNING, logger="nodestream.pipeline.extractors.files"
    ):
        assert await _records_from(archive_path) == [SIMPLE_RECORD]
    size_gb = f"{(limit + 1) / 1024**3:.1f} GB"
    limit_gb = f"{limit / 1024**3:.1f} GB"
    assert (
        "Skipping oversized member big.csv in "
        f"{archive_path} ({size_gb} > {limit_gb} limit, ZipArchive.max_member_bytes)"
    ) in caplog.text


@pytest.mark.asyncio
async def test_invalid_zip_logs_the_unpack_error(base_dir: str, mocker):
    path = Path(base_dir) / "bad.zip"
    path.write_bytes(b"this is not a zip file")
    subject = FileExtractor([LocalFileSource([path])])
    subject.logger = mocker.Mock()
    assert [record async for record in subject.extract_records()] == []
    message = _warning_message(subject.logger)
    assert message.startswith(f"Failed to read archive {path}.")
    assert message.endswith("File is not a zip file")


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("tables/one.sas7bdat", True),
        ("tables\\one.sas7bdat", True),
        ("/tables/one.sas7bdat", True),
        ("one.SAS7BDAT", True),
        ("notes/readme.txt", True),
        ("folder/", False),
        ("", False),
        ("__MACOSX/._one.sas7bdat", False),
        ("tables/._one.sas7bdat", False),
    ],
)
def test_is_data_member(name, expected):
    assert _is_data_member(name) is expected


def test_zip_is_an_archive_codec_not_a_file_format():
    assert ".zip" in SUPPORTED_ARCHIVE_FORMAT_REGISTRY
    assert ".zip" not in SUPPORTED_FILE_FORMAT_REGISTRY

    class PluginZip(FileCodec, alias=".zip"):
        def read_file_from_handle(self, reader):
            return []

    try:
        assert SUPPORTED_FILE_FORMAT_REGISTRY.get(".zip") is PluginZip
    finally:
        del SUPPORTED_FILE_FORMAT_REGISTRY.registry[".zip"]


@pytest.mark.asyncio
async def test_file_extractor_loads_format_plugins(base_dir: str, mocker):
    module_path = Path(base_dir) / "demo_format.py"
    module_path.write_text(
        "from nodestream.pipeline.extractors.files import FileCodec\n"
        "\n"
        "class DemoFormat(FileCodec, alias='.demoplug'):\n"
        "    def read_file_from_handle(self, reader, source=None):\n"
        "        return [{'body': reader.read().decode(), 'source': source}]\n",
        encoding="utf-8",
    )

    def load():
        spec = importlib.util.spec_from_file_location("demo_format_plugin", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    entrypoint = mocker.Mock()
    entrypoint.load.side_effect = load
    mocker.patch.object(FileCodec, "entrypoints", return_value=[entrypoint])
    path = Path(base_dir) / "sample.demoplug"
    path.write_text("hello", encoding="utf-8")
    try:
        assert await _records_from(path) == [{"body": "hello", "source": str(path)}]
        entrypoint.load.assert_called_once()
    finally:
        SUPPORTED_FILE_FORMAT_REGISTRY.registry.pop(".demoplug", None)
