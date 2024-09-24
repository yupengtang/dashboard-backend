import argparse
import os
import sys
from flask import Flask
from flask_cors import CORS
import pandas as pd
import warnings
from routes import add_routes
warnings.filterwarnings("ignore")
import platform


def create_app():
    app = Flask(__name__)  
    CORS(app, resources={r"/*": {"origins": "*"}})
    add_routes(app)
    return app


def start_server():
    parser = argparse.ArgumentParser()

    # API flag
    parser.add_argument(
        "--host",
        default="127.0.0.1",  # Default to localhost for local development
        help="The host to run the server",
    )
    parser.add_argument(
        "--port",
        default=8000,  # Default port for local development
        help="The port to run the server",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run Flask in debug mode",
    )

    args = parser.parse_args()

    # Check if running in a cloud environment with a specified PORT
    host = "0.0.0.0"  # Bind to 0.0.0.0 to allow external access
    port = int(os.environ.get("PORT", args.port))  # Use PORT from environment if available

    server_app = create_app()

    server_app.run(debug=args.debug, host=host, port=port)


if __name__ == "__main__":
    start_server()
