from flask import Blueprint

webcrawler_blueprint = Blueprint('webcrawler', __name__, template_folder='templates')

from . import views