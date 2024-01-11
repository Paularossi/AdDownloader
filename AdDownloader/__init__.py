"""Top-level package for AdDownloader."""
# AdDownloader/__init__.py
# import classes and methods from the package as a whole

__app_name__ = "AdDownloader"
__version__ = "0.1.0"

# define errors
(
    SUCCESS,
    DIR_ERROR,
    FILE_ERROR,
    DB_READ_ERROR,
    DB_WRITE_ERROR,
    JSON_ERROR,
    ID_ERROR,
) = range(7)

ERRORS = {
    DIR_ERROR: "config directory error",
    FILE_ERROR: "config file error",
    DB_READ_ERROR: "database read error",
    DB_WRITE_ERROR: "database write error",
    ID_ERROR: "to-do id error",
}



