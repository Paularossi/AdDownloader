from AdDownloader.app import app


def start_gui(server_kwargs: dict = None):
    app.run_server(**(server_kwargs or {}))
