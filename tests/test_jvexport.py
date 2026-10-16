from keibaai import jvexport


class FakeJV:
    """JV-Link の代わり。キーごとに JVRTOpen の戻り値を順に返し、開けたキーでは1件だけ読ませる。"""

    def __init__(self, opens: dict[str, list[int]]):
        self.opens = {k: list(v) for k, v in opens.items()}
        self.calls, self.closed, self.reading = [], 0, None

    def JVRTOpen(self, spec, key):
        self.calls.append(key)
        code = self.opens[key].pop(0)
        self.reading = [f"O1{key}"] if code == 0 else []
        return code

    def JVRead(self, buff, size, fname):
        if self.reading:
            return 1, self.reading.pop(0) + "\r\n", 0, "x.jvd"
        return 0, "", 0, ""

    def JVClose(self):
        self.closed += 1


def test_export_rt_retries_once_on_413(tmp_path, monkeypatch):
    monkeypatch.setattr(jvexport, "RT_INTERVAL", 0)
    monkeypatch.setattr(jvexport.time, "sleep", lambda s: None)
    jv = FakeJV({"A": [-413, 0], "B": [-413, -413], "C": [-1], "D": [0]})
    out = tmp_path / "RT_0B31.txt"
    assert jvexport.export_rt(jv, "0B31", ["A", "B", "C", "D"], out) == 2
    assert jv.calls == ["A", "A", "B", "B", "C", "D"]  # -413 だけ1回やり直す。2回目も -413 なら飛ばす。-1 はやり直さない
    assert out.read_text(encoding="utf-8").splitlines() == ["x.jvd\tO1A", "x.jvd\tO1D"]
