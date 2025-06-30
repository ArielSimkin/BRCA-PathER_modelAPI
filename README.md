# BRCA-PathER_modelAPI
An API based on a modified Xception model layout for ER classification from WSIs of breast cancer

Every file is responsible for different, verstile/specific activities:

Data_preperation.py - The part used for preprocessing, filtering and sorting the slides chosed for the dataset

Xception_prepreration.py - This part is taking the base model as Xcpetion, adding several modifications and an Attention CBAM block.
                           Then, The model is trained, tested and showing reasults for further examination.
                           
System_pipline.py - This part is used as the server side and the pipline through which the client's slide is tiled, preprocessed,
                    examined by the model and sends the information package back to the dlient for the use of the API builder.

client.py - This part is resposible for operating the client's side's API while comunicating with the server.

html_generator.py - This section is responsible for returning the HTML code needed for the viewer of the original/proccessed image as a .svs file.
