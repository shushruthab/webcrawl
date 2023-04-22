from . import main_blueprint
from flask import render_template, request, redirect, url_for, jsonify

@main_blueprint.route('/hello')
def hello():
    return jsonify({'success': 'true'})