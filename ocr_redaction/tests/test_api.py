import cv2
import pytest

pytest.importorskip("httpx")
pytest.importorskip("multipart")
from fastapi.testclient import TestClient  # noqa: E402

from api import main as api_main  # noqa: E402
from ocr_engine import InferenceEngine  # noqa: E402

from synthetic_docs import make_page  # noqa: E402


class FakeRecognizer:
    def recognize_batch(self, crops):
        return ["TEXT"] * len(crops), 0.0


@pytest.fixture()
def client(monkeypatch):
    engine = InferenceEngine(recognizer=FakeRecognizer())
    monkeypatch.setattr(api_main, "_engine", engine)
    return TestClient(api_main.app)


def _png_bytes():
    img, _ = make_page("light", 7)
    return cv2.imencode(".png", img)[1].tobytes()


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_ocr_roundtrip(client):
    r = client.post("/ocr", files={"file": ("page.png", _png_bytes(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["words"] and set(body["words"][0]) == {"word", "bounding_box"}
    red = client.get(body["redacted_image_url"])
    assert red.status_code == 200 and red.headers["content-type"] == "image/png"
    assert client.get(body["text_json_url"]).status_code == 200


def test_rejects_bad_inputs(client):
    assert client.post("/ocr", files={"file": ("a.txt", b"hi", "text/plain")}).status_code == 415
    assert client.post("/ocr", files={"file": ("a.png", b"not an image", "image/png")}).status_code == 400
    assert client.get("/ocr/doesnotexist/redacted").status_code == 404
