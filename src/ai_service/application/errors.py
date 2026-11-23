"""Application-level errors shared by ports and use cases."""


class BackendUnavailableError(RuntimeError):
    """Raised when a canonical backend dependency cannot be reached."""


class CatalogCategoryUnresolvedError(ValueError):
    """The requested category does not resolve to an active catalog category."""


__all__ = ["BackendUnavailableError", "CatalogCategoryUnresolvedError"]
