# 5. Deploying Machine Learning Models

## 5.1 Intro / Session overview

In this session we'll use the model we made in session 4 for churn prediction.

This session contains the deployment of the model. If we want to use the model to predict new values without running the code. The way to use the model in different machines without running the code, is to deploy the model in a server (run the code and make the model). After deploying the code in a machine used as server we can make some endpoints (using api's) to connect from another machine to the server and predict values.

To deploy the model in a server there are some steps:

- After training, the model is saved for making predictions in the future (pickle file)
- Make the API endpoints in order to request predictions (using flask)
- Some other server deployment options (will be discussed from section 5 to 9)

## 5.2 Saving and Loading the Model

**In this section we cover the idea "How to use the trained model in future":**

- To save the built model, we can use pickle library:
  - `Pickle` is a Python built-in library but for some reason if we don't have it, we can install it using `pip install pickle-mixin`
  - After training the model and being the model ready for the prediction process, we use the following code to save the model for later:
    - ```python
      import pickle
      with open('model.bin', 'wb') as f_out:
          pickle.dump((dv, model), f_out)
      ```
    - In the code above we are making a binary file named **model.bin** and writing the *DictVectorizer* for one hot encoding and *model* as an array. (We save the file as binary to write bytes instead of text)
  - To be able to use the model in future without running the code, we need to read the binary file we saved previously:
    - ```python
      with open('mode.bin', 'rb') as f_in:  ## Note that never open a binary file you do not trust!
          dv, model = pickle.load(f_in)
      ```
  - With unpacking the model and the DictVectorizer, we're able to making prediction for the new input values without training the model.

## 5.3 Web Services: Introduction to Flask

This section is about what is a web service and how to create a simple web service.

- What is actually a web service?
  - A web service is a method used to communicate between electronic devices
  - There are some methods in web services we can use it to satisfy our problems. Below we would list some of them:
    - **GET**: GET is a method used to retrieve files. For example, when we search for a cat image in google we actually request cat images with GET method
    - **POST**: POST is the second common method used in web services. For example, in a sign up form, when we submit our name, username, passwords, etc. We post our data to a server that is using the web service. (Note that there is no specification where the data goes)
    - **PUT**: PUT is same as POST but we specify where the data goes to store
    - **DELETE**: DELETE is a method that is used to request to delete some data from the server
    - We can read more about HTTP methods from [here](https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods)
  - There are plenty of libraries available, almost in every language to create a simple web service. For this project we'll use Flask library from Python.
    - We can install the library with `pip install flask`
    - We can run the sample code below to create a simple web service:
      - ```python
           from flask import Flask
           app = Flask('churn-app') # give an identity to our web service
           @app.route('/ping', methods=[GET])
           def ping():
               return 'PONG'

           if __name__=='__main__':
              app.run('debug=True, host='0.0.0.0', port=9696) # run the code in local machine with the debugging mode True and port 9696
          ```
      - With the code above we made a simple web server and created a route named ping that would send pong string
      - To test the app, we need to search for `localhost:9696/ping` in our browser. We'll see the 'PONG' string is received.
    - To use our web server to predict new values we must modify our code base. We'll do that in the next section.

## 5.4 Serving the Churn Model with Flask

