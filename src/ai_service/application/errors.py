"""Application-level errors shared by ports and use cases."""


class BackendUnavailableError(RuntimeError):
    """Raised when a canonical backend dependency cannot be reached."""


__all__ = ["BackendUnavailableError"]
