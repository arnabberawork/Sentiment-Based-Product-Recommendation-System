# Import required libraries and modules
from flask import Flask, render_template, request, jsonify
import model  # Importing the model module

# Initialize Flask app
app = Flask(__name__)

# Predefined list of valid user IDs
valid_userid = [
    '00sab00', '1234', 'zippy', 'zburt5', 'joshua', 'dorothy w',
    'rebecca', 'walker557', 'samantha', 'raeanne', 'kimmie',
    'cassie', 'moore222'
]

# Route for the home page
@app.route('/')
def view():
    return render_template('index.html')

# Route to handle product recommendations using JSON input
@app.route('/recommend', methods=['POST'])
def recommend_top5():
    data = request.get_json()

    if not data or 'User Name' not in data:
        return jsonify({'error': 'Missing User Name'}), 400

    user_name = data['User Name']

    # If user is valid, generate and return top 5 recommendations
    if user_name in valid_userid:
        top20_products = model.recommend_products(user_name)
        get_top5 = model.top5_products(top20_products)

        # Return recommendations as JSON
        return jsonify({
            'message': 'Recommended products',
            'columns': list(get_top5.columns),
            'data': get_top5.values.tolist()
        })

    # Handle invalid user input
    return jsonify({'message': 'No Recommendation found for the user'}), 404

# Run the app
if __name__ == '__main__':
    app.debug = False
    app.run()
