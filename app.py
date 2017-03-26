""""
    app.py

    Starts the Flask server.
"""

from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['search']

    return render_template('search.html')

if __name__ == "__main__":
    app.run(debug=True)
