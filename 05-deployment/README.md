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
      with open('mode.bin', 'rb') as f_in:  ## Note that never open a binary file we do not trust!
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

In this section we'll implement the functionality of prediction to our churn web service and how to make it usable in development environment.

- To make the web service predict the churn value for each customer we must modify the code in section 3 with the code we had previously. Below we can see how the code works in order to predict the churn value.
- In order to predict we first need to load the previously saved model and use a prediction function in a special route.
  - To load the previously saved model we use the code below:
    - ```python
      import pickle

      with open('churn-model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)
      ```
  - To predict the value for a customer we need a function like below:
    - ```python
      def predict_single(customer, dv, model):
        X = dv.transform([customer])  ## apply the one-hot encoding feature to the customer data 
        y_pred = model.predict_proba(X)[:, 1]
        return y_pred[0]
      ```
  - Then at last we make the final function used for creating the web service:
    - ```python
      @app.route('/predict', methods=['POST'])  ## in order to send the customer information we need to post its data.
      def predict():
      customer = request.get_json()  ## web services work best with json frame, So after the user post its data in json format we need to access the body of json.

      prediction = predict_single(customer, dv, model)
      churn = prediction >= 0.5

      result = {
          'churn_probability': float(prediction), ## we need to convert numpy data into python data for flask framework
          'churn': bool(churn),  ## same as the line above, converting the data using bool method
      }

      return jsonify(result)  ## send back the data in json format to the user
      ```
  - The whole code above is available in [predict.py](https://github.com/MuhammadAwon/ml-engineering/blob/main/05-deployment/code/predict.py)
  - At last we run the code to see the result. It can't use a simple request in web browser, therefore, we can run the following code to post a new user data and see the response:
    - ```python
      ## a new customer informations
      customer = {
        'customerid': '8879-zkjof',
        'gender': 'female',
        'seniorcitizen': 0,
        'partner': 'no',
        'dependents': 'no',
        'tenure': 41,
        'phoneservice': 'yes',
        'multiplelines': 'no',
        'internetservice': 'dsl',
        'onlinesecurity': 'yes',
        'onlinebackup': 'no',
        'deviceprotection': 'yes',
        'techsupport': 'yes',
        'streamingtv': 'yes',
        'streamingmovies': 'yes',
        'contract': 'one_year',
        'paperlessbilling': 'yes',
        'paymentmethod': 'bank_transfer_(automatic)',
        'monthlycharges': 79.85,
        'totalcharges': 3320.75
      }
      import requests ## to use the POST method we use a library named requests
      url = 'http://localhost:9696/predict' ## this is the route we made for prediction
      response = requests.post(url, json=customer) ## post the customer information in json format
      result = response.json() ## get the server response
      print(result)
      ```
- Until here we have seen how we make a simple web server that predicts the churn value for every user. When we run the app, we'll see a warning that it is not a **WGSI server** and not suitable for production environements. To fix this issue and run this as a production server there are plenty of ways available:
  - One way to create a WSGI server is to use `gunicorn`. To install it use the command `pip install gunicorn` and to run the WGSI server we can simply run it with the command `gunicorn --bind 0.0.0.0:9696 predict:app`. Note that in **predict:app** the name predict is the name we set for our file containing the code `app = Flask('churn')` (for instance, in our case it's predict.py). We may need to change it to whatever we named our Flask app file.
  - Windows users may not be able to use gunicorn library because windows system do not support some dependencies of the library. So to be able to run this on windows machine, there is an alternative library `waitress` and we can install it using the command `pip install waitress`.
  - To run the waitress wgsi server, use the command `waitress-serve --listen=0.0.0.0:9696 predict:app`. After running this command we should be able to get the same result as gunicorn.
- So until here we are able to make a production server that predicts the churn value for new customers. In the next section we'll see how to solve library version conflictions in each machine and manage the dependencies for production environments.

## 5.5 Python Virtual Environment: Pipenv

This section explains how to make virtual environment for our project. Let's understand what is a virtual environment and how to make it:

- Everytime we're running a file from a directory, we're using the executive files from a global directory. For example, when we install Python on our machine the executable files that are able to run our codes will go to somewhere like */home/username/python/bin* and `pip` command may go to */home/username/python/bin/pip*.
- Sometimes the versions of libraries conflict (the project may not run or it gets into massive errors). For example, we have an old project that uses sklearn library with the version of `0.24.1` and now we want to run another project using sklearn version `1.0.0`. We may get into errors because of the version conflict:
  - To solve the conflict we can make virtual environments. Virtual environment is something that can separate the libraries installed in our system and the libraries with specified version we want our projects to run with. There are a lot of ways to create a virtual enironments. One of them are using `pipenv` library.
  - pipenv is a library that can create a virtual enironment. To install this library just use the classic method `pip install pipenv`.
  - After installing pipenv we can install the libraries we want for our project in the new virtual environment. Which can be done with `pipenv install numpy sklearn==0.24.1 flask`.
  - Note that using the pipenv command we make two files names *Pipfile* and *Pipfile.lock*. If we look into these files closely we can see that in Pipfile the libraries we installed are named. If we specified a library with its version, it's also specified in Pipfile.
  - In *Pipfile.lock* we can see that each library with it's installed version is named and a hash file is there to reproduce if we move the environment to another machine.
  - If we want to run the project in another machine, we can easily installed the libraries we want with the command `pipenv install`. This command will look into *Pipfile* and *Pipfile.lock* to install the libraries with specified version.
  - After installing the required libraries we can run the project in the virtual environment with `pipenv shell` command. This will go to the virtual environment's shell and then any command we execute will use the virtual environment's libraries.
- Until here we have made a virtual enivornment for our libraries with a required specified version. To separate this environment even more, such as making gunicorn be able to run in windows machines we need another way, which is using Docker. Docker allows us to separate everything more than creating usual virtual enironment way and make any project able to run on any machine that supports Docker smoothly.

## 5.6 Environment Management: Docker

### Installing Docker

To isolate the project files from our system machine, we can use Docker. With Docker we are able to pack everything our project contains in the system that we want to run in any system machine. For example, if we want Ubuntu 20.4 we can have it in a mac or windows machine or other operating systems like linux.

To get started with Docker for the churn prediction project we can follow the following instructions:

**Docker Desktop (Ubuntu)**

```
sudo apt-get install docker.io
```

To run docker without `sudo`, follow these [instructions](https://docs.docker.com/engine/install/linux-postinstall/).

**Docker Desktop (Windows)**

To install the Docker on windows we can just follow the instructions by Andrew Lock in this [link](https://andrewlock.net/installing-docker-desktop-for-windows/).

- Once our project is packed in a Docker container, we're able to run the project on any machine.
- First we have to make a Docker image. The Docker image has the settings and dependencies we have in our project. To find Docker images that we need we can simply search the [Docker Hub](https://hub.docker.com/search?type=image).

To run the python in the terminal, we need to entry `docker run -it --rm python:3.8.11-slim` command, where:
  - `-it` to access python docker image in the terminal
  - `--rm` remove the docker image after exit

We can also get inside the python image by adding `--entrypoint=base`. This command will activate bash using the python docker image, like so:
  - `docker run -it --rm --entrypoint=base python:3.8.11-slim`

Here's what our Dockerfile contains (note: there should be no comments in Dockerfile, therefore, we need to remove the comments when we copy the code below):

```
# First install the base image (i.e., python 3.8), the slim version have less size
FROM python:3.8.12-slim

# Install pipenv library in Docker 
RUN pip install pipenv

# We have created a directory in Docker named app and we're using it as our working directory 
WORKDIR /app                                                                

# Copy the Pip files into our working derectory 
COPY ["Pipfile", "Pipfile.lock", "./"]

# Install the pipenv dependencies we had from the project and deploy them 
RUN pipenv install --deploy --system

# Copy any python files and the model we had to the working directory of Docker 
COPY ["*.py", "model_C=1.0.bin", "./"]

# We need to expose the 9696 port because we're not able to communicate with Docker outside it
EXPOSE 9696

# If we run the Docker image, we want our churn app to be running
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
```

If we don't put the last line `ENTRYPOINT`, we will be in a python shell. Note that for the entrypoint, we put our commands in double quotes.

After creating the Dockerfile, we need to build it:
```
docker build -t churn-prediction . # dot means cwd
```
  - We use `-t` flag for specifying the tag name which is `churn-prediction` in our case.

To run it, execute the command below:

```
docker run -it -p 9696:9696 churn-prediction:latest
```

Here we use:

-  `-it` in order to the Docker run from terminal and shows the result.
-  `-p` to map the 9696 port of the Docker to 9696 port of our machine. (First 9696 is the port number of our machine and the last one is Docker container port).

At last we've deployed the prediction app inside a Docker container.

## 5.7 Deployment to the Cloud: AWS Elastic Beanstalk (optional)

To create AWS Elastic Beanstalk app, we need an AWS account if we don't have already. Instructions to create an account on AWS can be found [here](https://mlbookcamp.com/article/aws).

Elastic Beanstalk helps us deploy and manage web applications effieciently using load balancer. Which means it scales up or down according to the traffic demand.

We need to install awsebcli tool to run our application on the AWS cloud. Following are the instructions to deploy the app with Elastic Beanstalk:

- Install awsebcli using `pipenv install awsebcli --dev`
  - This will install awsebcli only for development.
- To initialize the project name (e.g. `churn-serving`) to run on docker platform in the specified region, we use the command `eb init -p docker -r eu-west-1 churn-serving`
  - The above command will create a folder `./elasticbeanstalk` that contains a `config.yml` file.
- To test the app works fine locally we use the command `eb local run --port 9696`
  - This command will re-run `RUN pipenv install --system --deploy` from Dockerfile because we have updated Pipfile which includes awsebcli package.
- Running `python predict-test.py` should return us the prediction.
- To run our app on the cloud we'll need to create elastic beanstalk serving environment, which can be made with `eb create churn-serving-env`
  - The host link will appear on the terminal, for instance: `churn-serving-env.eba-gg58yj4v.eu-west-1.elasticbeanstalk.com`
- Next, we'll need to add the above link in `predict-test.py` file by replacing the url. We don't need to specify the port because elastic beanstalk will automatically map its port 80 with the app at port 9696. Below is the code block how this step will look like in the predict-test script:
  - predict-test.py 
    ```python
    # eb host link
    host = 'churn-serving-env.eba-gg58yj4v.eu-west-1.elasticbeanstalk.com'

    # app url
    url = f'http://{host}/predict'
    ```
- If everything is setup properly then running the file `predict-test.py` should return the customer prediction in the terminal.

It is important to stop the eb server if we don't need it, otherwise it will keep adding the cost. Running `eb terminate churn-serving-env` in the terminal will stop the server.

**Note**: The host link will be open to anyone and it can be used to send request. We should turn it down if we don't want it open to the entire public.

## 5.8 Summary

In the session five we learned the following topics:

- We learned how to save the model and load it to re-use the trained model to make predictions on new data.
- How to deploy the model in a web service.
- How to create a virtual environment.
- How to create a container and run our code in any operating system.
- How to implement our code in a public web service and access it from outside a local computer.

In the next session we will learn the algorithms such as Decision trees, Random forests and Gradient boosting as an alternative way of combining decision trees.

## 5.9 Explore More

- Flask is not the only framework for creating web services. Try others, e.g., FastAPI
- Experiment with other ways for managing environment, e.g., virtual env, conda, poetry.
- Explore other ways of deploying web services, e.g. [GCP](https://cloud.google.com/gcp), [Azure](https://azure.microsoft.com/en-us/), [Heroku](https://www.heroku.com/), [PythonAnywhere](https://www.pythonanywhere.com/) etc.

## Deployment Tutorials

- [Using PythonAnywhere to host the Python Web App for free!! (an alternative to AWS/Azure/Google cloud)](https://github.com/nindate/ml-zoomcamp-exercises/blob/main/how-to-use-pythonanywhere.md)
- [Deploy Churn Service on Heroku](https://github.com/razekmaiden/churn_service_heroku.git)
- [Installing and using pipenv and Docker on Intel Macs](https://github.com/ziritrion/ml-zoomcamp/blob/main/notes/05b_virtenvs.md)
- [Azure Guide](https://github.com/yusyel/azure#1-creating-azure-account)
