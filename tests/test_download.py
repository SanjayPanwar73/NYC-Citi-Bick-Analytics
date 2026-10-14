import io
import json
import os
import unittest
import zipfile
from pathlib import Path
import tempfile
import _env  # noqa: F401
import requests
import download_data as dd


def make_zip(files):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w", zipfile.ZIP_STORED) as z:
        for n, c in files.items():
            z.writestr(n, c)
    return b.getvalue()


class Resp:
    def __init__(self, data, status, fail_after=None):
        self.data, self.status_code, self.fail_after = data, status, fail_after
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def raise_for_status(self): pass
    def iter_content(self, n):
        sent = 0
        for i in range(0, len(self.data), 100):
            if self.fail_after is not None and sent >= self.fail_after:
                raise requests.exceptions.ConnectionError("connection dropped")
            yield self.data[i:i + 100]; sent += 100


class Session:
    """Serves `data`; scripted: first call drops the connection after `drop_after` bytes."""
    def __init__(self, data, drop_first_after=None, ignore_range=False):
        self.data, self.drop, self.ignore_range, self.calls = data, drop_first_after, ignore_range, []
    def get(self, url, headers=None, stream=True, timeout=0, **k):
        rng = (headers or {}).get("Range")
        self.calls.append(rng)
        start = int(rng.split("=")[1].rstrip("-")) if rng and not self.ignore_range else 0
        fail = self.drop if len(self.calls) == 1 else None
        return Resp(self.data[start:], 206 if start else 200, fail)


class Download(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.data = make_zip({"202401-citibike-tripdata_1.csv": "a,b\n" + os.urandom(4000).hex()})
        self.dest = self.tmp / "202401-citibike-tripdata.zip"

    def test_resume_after_dropped_connection(self):
        s = Session(self.data, drop_first_after=600)
        size = dd.download_file("u", self.dest, len(self.data), session=s, sleep=lambda _: None)
        self.assertEqual(size, len(self.data))
        self.assertTrue(dd.is_complete(self.dest, len(self.data)))
        self.assertIsNotNone(s.calls[1])                           # second call used a Range header
        self.assertFalse(self.dest.with_name(self.dest.name + ".part").exists())
        self.assertEqual(json.loads(self.dest.with_name(self.dest.name + ".ok").read_text())["size"], len(self.data))

    def test_server_ignores_range_restarts_cleanly(self):
        s = Session(self.data, drop_first_after=600, ignore_range=True)
        dd.download_file("u", self.dest, len(self.data), session=s, sleep=lambda _: None)
        dd.validate_zip(self.dest)

    def test_truncated_never_marked_complete(self):
        s = Session(self.data[:-50])                                # server always short
        with self.assertRaises(RuntimeError):
            dd.download_file("u", self.dest, len(self.data), session=s, retries=2, sleep=lambda _: None)
        self.assertFalse(self.dest.exists())
        self.assertFalse(self.dest.with_name(self.dest.name + ".ok").exists())

    def test_corrupt_zip_of_correct_size_rejected(self):
        bad = bytearray(self.data); bad[len(bad) // 2] ^= 0xFF; bad = bytes(bad)
        with self.assertRaises(RuntimeError):
            dd.download_file("u", self.dest, len(bad), session=Session(bad), retries=2, sleep=lambda _: None)
        self.assertFalse(self.dest.exists())

    def test_html_error_page_rejected(self):
        page = b"<html>AccessDenied</html>" * 20
        with self.assertRaises(RuntimeError):
            dd.download_file("u", self.dest, len(page), session=Session(page), retries=1, sleep=lambda _: None)

    def test_existing_file_without_marker_is_revalidated(self):
        self.dest.write_bytes(self.data[:-10])                      # leftover partial from an old run
        self.assertFalse(dd.is_complete(self.dest, len(self.data)))
        self.assertTrue(self.dest.with_name(self.dest.name + ".corrupt").exists())   # kept, not deleted
        self.dest.write_bytes(self.data)
        self.assertTrue(dd.is_complete(self.dest, len(self.data)))

    def test_repeated_corruption_keeps_all_evidence(self):
        """Windows rename() cannot overwrite: a second corrupt file must get a new name, not crash."""
        for i in range(3):
            self.dest.write_bytes(b"bad%d" % i)
            dd.quarantine(self.dest)
        names = sorted(p.name for p in self.tmp.iterdir())
        self.assertEqual(names, [self.dest.name + ".corrupt", self.dest.name + ".corrupt.1", self.dest.name + ".corrupt.2"])
        bad = bytearray(self.data); bad[len(bad) // 2] ^= 0xFF; bad = bytes(bad)
        with self.assertRaises(RuntimeError) as cm:           # corruption detected twice in one call
            dd.download_file("u", self.dest, len(bad), session=Session(bad), retries=2, sleep=lambda _: None)
        self.assertIn("CRC", str(cm.exception))                # real cause reported, not a rename error

    def test_wanted_and_extraction(self):
        w = dd.wanted
        self.assertTrue(w("202402-citibike-tripdata.zip") and w("2024-citibike-tripdata.zip"))
        self.assertFalse(w("202405-citibike-tripdata.zip") or w("JC-202402-citibike-tripdata.csv.zip") or w("2023-citibike-tripdata.zip"))
        inner = make_zip({"202403-citibike-tripdata.csv": "c", "__MACOSX/._x.csv": "j"})
        z = zipfile.ZipFile(io.BytesIO(make_zip({"y/202401-a.csv": "a", "y/202402-a_1.csv": "b", "y/202402-a_2.csv": "b2",
                                                  "202403.zip": inner, "y/202405-a.csv": "late"})))
        out = self.tmp / "ext"; out.mkdir()
        names = dd.extract_csvs(z, out)
        self.assertEqual(sorted(names), ["202401-a.csv", "202402-a_1.csv", "202402-a_2.csv", "202403-citibike-tripdata.csv"])
        self.assertEqual(sorted(p.name for p in out.iterdir()), sorted(names))   # no stray .tmp files


if __name__ == "__main__":
    unittest.main()
