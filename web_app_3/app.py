from flask import Flask, render_template, request
from als_for_flask import for_flask
from content_based_for_flask import for_flask_content
from jinja2 import Template
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/user_rec', methods=['POST'])
def button1():
    user_name = (request.form['user_input'])
    best_num_player = request.form.getlist('check')
    min_time = (request.form['min_time'])
    max_time = (request.form['max_time'])
    df = for_flask(user_name, best_num_player, min_time, max_time)
    return render_template('user_rec.html', tables = df.to_html(index=False))

@app.route('/boardgame_rec', methods=['POST'])
def button2():
    board_game = (request.form['user_input'])
    best_num_player = request.form.getlist('check')
    min_time = (request.form['min_time'])
    max_time = (request.form['max_time'])
    df = for_flask_content(board_game, best_num_player, min_time, max_time)
    return render_template('boardgame_rec.html', tables = df.to_html(index=False))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
