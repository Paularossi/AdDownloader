from AdDownloader.app import app


def start_gui(server_kwargs: dict = {}):
    app.run_server(**server_kwargs)
