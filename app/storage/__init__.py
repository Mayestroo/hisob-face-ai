"""SQLite persistence boundary for local Face AI state."""

from app.storage.database import Database
from app.storage.schema import SCHEMA_VERSION, UnsupportedSchemaVersionError

__all__ = ["Database", "SCHEMA_VERSION", "UnsupportedSchemaVersionError"]
