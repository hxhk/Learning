from app import app

#routes
@app.route('/')
def hello():
    return "Hello,World!"
  
@app.route('/index')
def index():
    return "This is an index page..."

@app.route('/post/<int:post_id>')
def show_post(post_id):
    return 'Post id %d' % post_id

@app.route('/user/<username>')
def show_user_profile(username):
    return 'User %s' % username
