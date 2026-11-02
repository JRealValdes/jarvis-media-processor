"""Media inspection errors."""


class MediaError(Exception):
    """Base error for media operations."""


class FFprobeNotFoundError(MediaError):
    """FFprobe executable is missing or not runnable."""


class FFprobeError(MediaError):
    """FFprobe ran but failed to inspect the file."""

    def __init__(
        self,
        message: str,
        *,
        returncode: int | None = None,
        stderr: str | None = None,
    ) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr


class MediaFileNotFoundError(MediaError):
    """The requested media path does not exist."""
