from flask import Flask, request, jsonify, render_template_string, url_for
import requests

app = Flask(__name__, static_url_path='/static')

@app.route('/', methods=['GET'])
def upload_form():
    return render_template_string('''
        <html>
            <head>
                <link rel="stylesheet" type="text/css" href="{{ url_for('static', filename='style.css') }}">
                <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
                <script>
                    $(document).ready(function(){
                        $('form').on('submit', function(event){
                            $('.loading').show();
                        });
                        $('input[type="file"]').change(function(e){
                            var reader = new FileReader();
                            reader.onload = function(e) {
                                $('#display_image').attr('src', e.target.result);
                                $('#display_image').show();
                            }
                            reader.readAsDataURL(e.target.files[0]);
                        });
                    });
                </script>
            </head>
            <body>
               <center> <h1>Upload an image to predict pneumonia</h1></center>
               <center> <form action="/predict" method="post" enctype="multipart/form-data"></center>
                 <center> <input type="file" name="image" required><br></center>
                 <center> <input type="submit" value="Predict"></center>
                </form>
                <div class="loading" style="display:none;">
                    <p>Loading...</p>
                    <img src="{{ url_for('static', filename='loading.gif') }}" alt="Loading" />
                </div>
                <center> <img id="display_image" style="display:none; max-width: 100%; height: auto;"></center>
            </body>
        </html>
    ''')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided. Please provide an image file.'}), 400

    img_file = request.files['image']
    if img_file.filename == '':
        return jsonify({'error': 'No image selected for uploading'}), 400

    img_bytes = img_file.read()
    response = requests.post("http://localhost:5001/predict", files={'image': img_bytes})
    result = response.json()['prediction']
    return render_template_string('''
        <html>
            <head>
                <link rel="stylesheet" type="text/css" href="{{ url_for('static', filename='style.css') }}">
            </head>
            <body>
               <center> <h1>Prediction Result: {{ result }}</h1></center>
               <center>  <input type="button" value="Back" onclick="history.back()"></center> 
            </body>
        </html>
    ''', result=result)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
