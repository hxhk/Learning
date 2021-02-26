from app import app

#route
@app.route('/')
@app.route('/index')
def index():
    return "Hello,World!"
   
