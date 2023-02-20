# Import required modules
import re
from flask import Flask, jsonify
from flask import request
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from
import prosess_lstm
import prosess_nn
from tensorflow import keras
import json
import Nn_set
#hst_md_nn = None
app =  Flask(__name__)
app.json_encoder = LazyJSONEncoder
swagger_template = dict(
        info = {
            'title' : LazyString(lambda : 'API Documentation Kelompok 4 DSC-3 Platinum'),
            'version' : LazyString(lambda : '1.0.0'),
            'description' : LazyString(lambda : 'API Documentation Kelompok 4 DSC-3 Platinum'),
        },
        host = LazyString(lambda:request.host),

    )


swagger_config = {
        "headers" : [],
        "specs" : [
            {
                "endpoint" : 'docs',
                "route" : '/docs.json',
            }
        
        ],
        "static_url_path" : "/flasgger_static",
        "swager_ui" : True,
        "specs_route" : "/docs/",
       
       
    }
swagger = Swagger(app, template=swagger_template, config=swagger_config)



## -------------------------------------- LSTM ----------------------------------------------------
#End point insert text Neral Network
@swag_from("yml/lstm-file.yml", methods=['POST'])
@app.route('/file-text-lstm-for-training', methods=['POST'])
def upload_file_lstm():
    
    json_response={}
    uploaded_file = request.files['upfile_tweet']
    test_size = request.form.get('test_size')
    epochs = request.form.get('epochs')
    clen = request.form.get('cleaning')
    if uploaded_file.filename != '':
        
        # save the file
        uploaded_file.save(uploaded_file.filename)
        r = prosess_lstm.prosess(float(test_size),int(epochs),uploaded_file.filename,clen)
      
        #if 1==1 :
        #if type(r[0]) == keras.callbacks.History:
        if r != 0:
            
            json_response = {
                 'status_code' : 200,
                 'description' : "Training sukses",
                 'data' : { 'Rata -  Rata Acuracy ' : r}
                 
             }
        else:
            json_response = {
                 'status_code' : 201,
                 'description' : "training gagal",
                 'data' : "Tidak Error"
             }

    else: 
        json_response = {
                'status_code' : 400,
                'description' : "OK",
                'data' : 'Coba Lagi'
         }
        
    
    response_data = jsonify(json_response)
    return response_data

#End point insert raw text LSTM Long-Short Term Memory
@swag_from("yml/lstm-text.yml", methods=['POST'])
@app.route('/insert-text-lstm-for-testing', methods=['POST'])
def form_insert_raw_text_lstm():

    text = request.form.get('text').strip()
    clen = request.form.get('cleaning')
    sr =  prosess_lstm.testing_raw_text_lstm(text,clen)
    json_response = {
             'status_code' : 400,
             'description' :sr
         }
    
  
    
    response_data = jsonify(json_response)
    return response_data

## -------------------------------------- NN ----------------------------------------------------
#End point insert raw text Neral Network
@swag_from("yml/neural-text.yml", methods=['POST'])
@app.route('/e-test-insert-text-neural', methods=['POST'])
def form_insert_raw_text_neural():

    text = request.form.get('text').strip()
    clen = request.form.get('cleaning')
    sr =  prosess_nn.testing_raw_text_nn(text,clen)
   
    json_response = {
             'status_code' : 400,
             'description' : sr
         }
    
   
    
    response_data = jsonify(json_response)
    return response_data


#End point insert text Neral Network
@swag_from("yml/neural-file.yml", methods=['POST'])
@app.route('/afile-text-neural-for-training', methods=['POST'])
def upload_file_neural():
    #NeuralNethsts = Nn_set.Nn()
    json_response={}
    uploaded_file = request.files['upfile_tweet']
    test_size = request.form.get('test_size')
    #epochs = request.form.get('epochs')
    clen = request.form.get('cleaning')
    if uploaded_file.filename != '':
        
        # save the file
        uploaded_file.save(uploaded_file.filename)
        r = prosess_nn.prosess(float(test_size),int(epochs),uploaded_file.filename,clen)
        
        #print("tes")
        #print(type(r[0]))

        #if 1==1 :
        if type(r[0]) == keras.callbacks.History:
           
            json_response = {
                 'status_code' : 200,
                 'description' : "Training sukses",
                 'data' : r[1] #json.loads(r[1]) 
             }
        else:
            json_response = {
                 'status_code' : 200,
                 'description' : "training gagal",
                 'data' : "tidak ok"
             }

    else: 
        json_response = {
                'status_code' : 400,
                'description' : "OK",
                'data' : 'Try more'
         }
        
    
    response_data = jsonify(json_response)
    return response_data


if __name__ == '__main__':
    #app.run()
    app.run(debug=True)
    #app.run(host="localhost", port=8001, debug=True)