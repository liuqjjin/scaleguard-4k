"""Domain errors with actionable failure messages."""


class ScaleGuardError(RuntimeError):
    """Base error for expected pipeline failures."""


class ConfigurationError(ScaleGuardError):
    """Raised when a configuration is invalid or internally inconsistent."""


class ArtifactError(ScaleGuardError):
    """Raised when a worker returns a missing, ambiguous, or invalid artifact."""


class WorkerError(ScaleGuardError):
    """Raised when an upstream worker cannot complete its contract."""


class WorkerTimeoutError(WorkerError):
    """Raised when an upstream process exceeds its configured deadline."""


class UpstreamVerificationError(ScaleGuardError):
    """Raised when an upstream checkout differs from the audited lock."""
