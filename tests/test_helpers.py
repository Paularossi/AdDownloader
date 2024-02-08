"""Unit tests for helper functions."""
import pytest
from AdDownloader.helpers import NumberValidator, DateValidator, CountryValidator
from prompt_toolkit.validation import ValidationError
from prompt_toolkit.document import Document


def test_number_validator_valid():
    validator = NumberValidator()
    assert validator.validate(Document("123")) is None 


def test_number_validator_invalid():
    validator = NumberValidator()
    with pytest.raises(ValidationError):
        validator.validate(Document("abc"))


def test_date_validator_valid():
    validator = DateValidator()
    assert validator.validate(Document("2023-01-01")) is None


def test_date_validator_invalid():
    validator = DateValidator()
    with pytest.raises(ValidationError):
        validator.validate(Document("2023-02"))  # An invalid date


def test_country_validator_valid():
    validator = CountryValidator()
    assert validator.validate(Document("US")) is None


def test_country_validator_invalid():
    validator = CountryValidator()
    with pytest.raises(ValidationError):
        validator.validate(Document("XX")) 


# test_number_validator_valid()
# test_number_validator_invalid()
# test_date_validator_valid()
# test_date_validator_invalid()
# test_country_validator_valid()
# test_country_validator_invalid()
