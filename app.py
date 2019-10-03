from flask import Flask, escape, request
import json

app = Flask(__name__)

@app.route('/')
def hello():
    dataset = json.loads(open('src/database.json').read())

    
    return f'{dataset}'