from app import db
from sqlalchemy.dialects.postgresql import JSON

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True)
    email = db.Column(db.String(120), unique=True)
    password = db.Column(db.String(120), unique=True)

    def __init__(self, username, email, password):
        self.username = username
        self.email = email
        self.password = password

    def __repr__(self):
        return '<User {}>'.format(self.username)

class Data(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80))
    series_id = db.Column(db.String(100))
    date = db.Column(db.Date())
    value = db.Column(db.Float())


    def __init__(self, username, series_id, date, value):
        self.username = username
        self.series_id = series_id
        self.date = date
        self.value = value

    def __repr__(self):
        return '<User {}>'.format(self.username)