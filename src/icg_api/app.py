import io
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import RedirectResponse
from PIL import Image
from pydantic import BaseModel

from icg_api import __version__
from icg_api.captioner import Captioner
from icg_api.settings import Settings


class CaptionResponse(BaseModel):
    caption: str


class HealthResponse(BaseModel):
    status: str


class ErrorResponse(BaseModel):
    detail: str


def read_image(data: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(data))
        image.load()  # Decode now so truncated or corrupt files fail here, not in the model.
    except (OSError, Image.DecompressionBombError) as error:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "The uploaded file is not a readable image"
        ) from error
    return image


def create_app(settings: Settings | None = None, captioner: Captioner | None = None) -> FastAPI:
    """Build the API. The model is loaded on startup unless ``captioner`` is given."""
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.captioner = captioner or Captioner.from_files(
            settings.checkpoint_path,
            settings.vocabulary_path,
            device=settings.device,
            max_length=settings.max_caption_length,
        )
        yield

    app = FastAPI(
        title="ICG API",
        description="Generate image captions in the Albanian language.",
        version=__version__,
        lifespan=lifespan,
    )

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/docs")

    @app.get("/health")
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.post(
        "/predict",
        responses={
            status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
            status.HTTP_413_CONTENT_TOO_LARGE: {"model": ErrorResponse},
        },
    )
    def predict(
        request: Request,
        image: Annotated[UploadFile, File(description="Image to caption, e.g. JPEG or PNG")],
    ) -> CaptionResponse:
        """Generate an Albanian caption for the uploaded image."""
        data = image.file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(
                status.HTTP_413_CONTENT_TOO_LARGE,
                f"The image must be at most {settings.max_upload_bytes} bytes",
            )
        with read_image(data) as pil_image:
            caption = request.app.state.captioner.caption(pil_image)
        return CaptionResponse(caption=caption)

    return app
