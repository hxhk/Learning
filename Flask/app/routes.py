from app import app
from flask import request,render_template,jsonify
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
    
@app.route('/hello/<name>') #使用模板
def hello_world(name=None):
    return render_template('hello.html', name=name)
    
@app.route('/get', methods = ["GET"])   #get
def get_data():
	name = request.args.get("name")
	age = request.args.get("age")
	return jsonify({'name': name, "age":age})
	# return "name=%s, age=%d" % (name, age)
