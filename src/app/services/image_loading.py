"""Turns the accepted image source formats into decoded OpenCV images.

Accepted sources: local file path, http(s) URL, or base64 (optionally with a
``data:image/...;base64,`` prefix). URL download uses blocking I/O, so callers
must not run inside the event loop.
"""

import base64
import binascii
import os

import cv2
import numpy as np
import requests

from app.core.exceptions import ImageDownloadError, InvalidImageError

_DOWNLOAD_TIMEOUT_SECONDS = 30


def load_image(source: str) -> np.ndarray:
    """Decode an image from a file path, URL or base64 string.

    Args:
        source: The image in any accepted source format.

    Returns:
        The decoded image as a BGR numpy array.

    Raises:
        ImageDownloadError: If ``source`` is a URL that cannot be fetched.
        InvalidImageError: If the bytes cannot be decoded as an image.
    """
    if source.startswith(("http://", "https://")):
        image_bytes = _download(source)
    elif os.path.isfile(source):
        with open(source, "rb") as file:
            image_bytes = file.read()
    else:
        image_bytes = _decode_base64(source)

    # cv2.imdecode raises on an empty buffer and returns None for garbage.
    if not image_bytes:
        raise InvalidImageError("The provided image data is empty.")
    try:
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    except cv2.error as error:
        raise InvalidImageError(
            "The provided data could not be decoded as an image."
        ) from error
    if image is None:
        raise InvalidImageError("The provided data could not be decoded as an image.")
    return image


def is_local_file(source: str) -> bool:
    """Return whether ``source`` refers to an existing local file."""
    return not source.startswith(("http://", "https://")) and os.path.isfile(source)


def _download(url: str) -> bytes:
    try:
        response = requests.get(url, timeout=_DOWNLOAD_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as error:
        raise ImageDownloadError(
            f"Could not download image from URL: {error}"
        ) from error
    return response.content


def _decode_base64(data: str) -> bytes:
    # Tolerate standard data-URI payloads like "data:image/jpeg;base64,...".
    if data.startswith("data:") and "," in data:
        data = data.split(",", 1)[1]
    try:
        # Lenient decoding (non-alphabet characters ignored) matches what the
        # legacy API accepted; undecodable bytes are caught by cv2.imdecode.
        return base64.b64decode(data)
    except (binascii.Error, ValueError) as error:
        raise InvalidImageError(
            "Image is not a readable file path, URL or valid base64 string."
        ) from error
