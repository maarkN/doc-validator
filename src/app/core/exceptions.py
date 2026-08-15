"""Domain exceptions raised by services and translated to HTTP once, globally."""


class DomainError(Exception):
    """Base class for expected application failures.

    Attributes:
        code: Stable machine-readable identifier exposed in error responses.
        status_code: HTTP status the API layer maps this error to.
    """

    code = "internal_error"
    status_code = 500

    def __init__(self, detail: str) -> None:
        """Store the human-readable detail exposed in the error response."""
        super().__init__(detail)
        self.detail = detail


class InvalidImageError(DomainError):
    """The provided image could not be decoded or contains no usable face."""

    code = "invalid_image"
    status_code = 400


class ImageDownloadError(DomainError):
    """An image URL could not be fetched."""

    code = "image_download_failed"
    status_code = 400
