class MsenseError(Exception):
    """
    Base class for all mSense errors.
    """


class EvaluationFailure(MsenseError):
    """
    Raised by Discipline._eval when an evaluation cannot produce outputs.
    """

    def __init__(self, message: str, details: str = None) -> None:
        self.details = details
        super().__init__(message)


class DriverCapabilityError(MsenseError):
    """
    Raised when a problem formulation requires something the driver cannot do,
    e.g. a multi-objective problem handed to a single-objective driver.
    """
