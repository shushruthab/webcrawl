import os
from . import webcrawler_blueprint
from .utilities import crawl, prep_df, answer_question
import openai
from flask import Flask, redirect, render_template, request, url_for

processed_df_path = 'app/webcrawler/processed/embeddings.csv'
domain = "mountainviewbrewing.ca"
full_url = "https://mountainviewbrewing.ca/"

crawl(full_url)
df = prep_df(domain)

@webcrawler_blueprint.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        question = request.form["question"]
        result = answer_question(question=question)
        return redirect(url_for("webcrawler.index", result=result))

    result = request.args.get("result")
    return render_template("webcrawler/index.html", result=result)