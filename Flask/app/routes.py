from app import app

#routes
@app.route('/')
@app.route('/index')
def index():
    return "Hello,World!"
   
