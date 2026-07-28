from __future__ import annotations

import os

from flask import Flask

from config import Config
from extensions import cors, db, migrate
from models import Flow, Invitation, Organization, OrganizationFlowAccess, Role, User
from routes import api

def create_app(config_object: type[Config] | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_object(config_object or Config)

    db.init_app(app)
    migrate.init_app(app, db)
    cors.init_app(app)

    app.register_blueprint(api)

    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
