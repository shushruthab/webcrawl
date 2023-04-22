import os
from . import webcrawler_blueprint
from .utilities import crawl, prep_df, create_context, answer_question
import openai
from flask import Flask, redirect, render_template, request, url_for

domain ="seekingalpha.com"
full_url = "https://seekingalpha.com/"
crawl(full_url)
df = prep_df(domain)

@webcrawler_blueprint.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        question = request.form["question"]
        result = answer_question(df, model="ada", question=question, max_len=1800, size="ada", debug=False,max_tokens=150, stop_sequence=None)
        return redirect(url_for("webcrawler.index", result=result))

    result = request.args.get("result")
    return render_template("webcrawler/index.html", result=result)