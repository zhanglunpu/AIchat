from flask import Flask, render_template, request, jsonify
from app import app

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test3d')
def test_3d():
    return render_template('test_3d.html') 