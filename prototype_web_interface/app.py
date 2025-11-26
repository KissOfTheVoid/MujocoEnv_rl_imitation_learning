from flask import Flask, request, jsonify, send_file, render_template
import csv
import uuid

app = Flask(__name__)

PRODUCTS = [
    {"product_name": "green_cube", "product_id": 0},
    {"product_name": "red_cube", "product_id": 1},
]

cart = []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/products', methods=['GET'])
def get_products():
    return jsonify(PRODUCTS)

@app.route('/cart/add', methods=['POST'])
def add_to_cart():
    data = request.json
    product_id = data.get('product_id')
    product = next((p for p in PRODUCTS if p['product_id'] == product_id), None)
    if not product:
        return jsonify({"error": "Product not found"}), 404
    cart.append(product)
    return jsonify({"message": "Product added"}), 201

@app.route('/order', methods=['POST'])
def order():
    if not cart:
        return jsonify({"error": "Cart is empty"}), 400

    filename = f"order_{uuid.uuid4()}.csv"
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['product_name', 'product_id'])
        for item in cart:
            writer.writerow([item['product_name'], item['product_id']])

    cart.clear()
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
