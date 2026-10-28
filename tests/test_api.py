import pytest
from fastapi.testclient import TestClient
from PIL import Image

from conftest import image_bytes
from icg_api.app import create_app
from icg_api.captioner import CheckpointError
from icg_api.settings import Settings


class StubCaptioner:
    def __init__(self):
        self.images: list[Image.Image] = []

    def caption(self, image: Image.Image) -> str:
        self.images.append(image)
        return "Një burrë me këmishë të zezë po këndon."


@pytest.fixture
def captioner() -> StubCaptioner:
    return StubCaptioner()


@pytest.fixture
def client(captioner):
    app = create_app(Settings(max_upload_bytes=50_000), captioner=captioner)
    with TestClient(app) as client:
        yield client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_redirects_to_docs(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


@pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
def test_predict_returns_caption(client, captioner, fmt):
    response = client.post("/predict", files={"image": ("photo", image_bytes(fmt=fmt))})
    assert response.status_code == 200
    assert response.json() == {"caption": "Një burrë me këmishë të zezë po këndon."}
    assert captioner.images[0].size == (64, 48)


def test_predict_requires_image_field(client):
    response = client.post("/predict", files={"file": ("photo.png", image_bytes())})
    assert response.status_code == 422


def test_predict_rejects_non_images(client, captioner):
    response = client.post("/predict", files={"image": ("notes.txt", b"not an image")})
    assert response.status_code == 400
    assert captioner.images == []


def test_predict_rejects_truncated_images(client):
    response = client.post("/predict", files={"image": ("photo.png", image_bytes()[:500])})
    assert response.status_code == 400


def test_predict_rejects_oversized_uploads(client):
    response = client.post("/predict", files={"image": ("big.png", b"\0" * 50_001)})
    assert response.status_code == 413


def test_startup_fails_when_weights_are_missing(tmp_path):
    pointer = tmp_path / "model.pth.tar"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\n")
    app = create_app(Settings(checkpoint_path=pointer))
    with pytest.raises(CheckpointError), TestClient(app):
        pass
