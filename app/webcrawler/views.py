import os
from . import webcrawler_blueprint
from .utilities import crawl, prep_df, answer_question,check_namespace
import openai
import pinecone
from flask import Flask, redirect, render_template, request, url_for, jsonify
from dotenv import dotenv_values

# processed_df_path = 'app/webcrawler/processed/embeddings.csv'

domain = "caffemira.com/"
full_url = "https://www.caffemira.com/"
if check_namespace(domain=domain):
    pass
else:
    crawl(full_url)
    df = prep_df(domain)

@webcrawler_blueprint.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        data = request.get_json()
        question = data.get("message", "")
        message = answer_question(question=question, domain=domain)
        return jsonify({'answer': message})
    return render_template("webcrawler/chat.html")

@webcrawler_blueprint.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        question = request.form["question"]
        result = answer_question(question=question)
        return redirect(url_for("webcrawler.index", result=result))

    result = request.args.get("result")
    return render_template("webcrawler/index.html", result=result)