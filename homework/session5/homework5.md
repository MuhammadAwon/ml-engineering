## Homework

In this homework, we'll use Credit Card Data from the previous homework.

## Question 1

- Install Pipevn
- What's the version of pipenv we installed?
- Use `--version` of find out

## Question 2

- Use Pipenv to install Scikit-Learn version 1.0.2
- What's the first hash for scikit-learn we get in Pipfile.lock?

Note: We should create an empty folder for homework and do it there.

### Models

We've prepared a dictionary vectorizer and a model.

They were trained (roughly) using this code:

```python
features = ['reports', 'share', 'expenditure', 'owner']
dicts = df[features].to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X = dv.fit_transform(dicts)

model = LogisticRegression(solver='liblinear').fit(X, y)
```

> **Note**: We don't need to train the model. This code is just for our reference.

And then saved the Pickle. Download them:

* DictVectorizer
* LogisticRegression

With `wget`:

```bash
PREFIX=https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/course-zoomcamp/cohorts/2022/05-deployment/homework
wget $PREFIX/model1.bin
wget $PREFIX/dv.bin
```

## Question 3

Let's use these models!

- Write a script for loading these models with pickle
- Score this client:

```json
{"reports": 0, "share": 0.001694, "expenditure": 0.12, "owner": "yes"}
```

What's the probability that this client will get a credit card?

- 0.162
- 0.391
- 0.601
- 0.993

If we're getting errors when unpickling the files, check their checksum:

```bash
$ md5sum model1.bin dv.bin
3f57f3ebfdf57a9e1368dcd0f28a4a14  model1.bin
6b7cded86a52af7e81859647fa3a5c2e  dv.bin
```

## Question 4

Now let's serve this model as a web service.

- Install Flask and gunicorn (or waitress, if we're on Windows)
- Write Flask code for serving the model
- Now score this client using `requests`:

```python
url = "YOUR_URL"
client = {"reports": 0, "share": 0.245, "expenditure": 3.438, "owner": "yes"}
requests.post(url, json=client).json()
```

What's the probability that this client will get a credit card?

- 0.274
- 0.484
- 0.698
- 0.928

### Docker

Install [Docker](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/05-deployment/06-docker.md). We'll use it for the next two questions.

For these questions, we prepared a base image: `svizor/zoomcamp-model:3.9.12-slim`. We'll need to use it (see Question 5 for an example).

This image is based on `python:3.9.12-slim` and has a logistic regression model (a different one) as well a dictionary vectorizer inside.

This is how the Dockerfile for this image looks like:

```docker
FROM python:3.9.12-slim
WORKDIR /app
COPY ["model2.bin", "dv.bin", "./"]
```

We already built it and then pushed it to `[svizor/zoomcamp-model:3.9.12-slim](https://hub.docker.com/r/svizor/zoomcamp-model)`.

> **Note**: We don't need to build this docker image, it's just for our reference.

## Question 5

Download the base image `svizor/zoomcamp-model:3.9.12-slim`. We can easily make it by using [docker pull](https://docs.docker.com/engine/reference/commandline/pull/) command.

So what's the size of this base image?

- 15 Mb
- 125 Mb
- 275 Mb
- 415 Mb

We can get this information when running `docker images` - it'll be in the "SIZE" column.

### Dockerfile

Now create our own Dockerfile based on the image we prepared.

It should start like that:

```docker
FROM svizor/zoomcamp-model:3.9.12-slim
# add your stuff here
```

Now complete it by adding following steps:

- Install all dependencies from the Pipenv file
- Copy our Flask script
- Run it with gunicorn

After that, we can build our docker image.

## Question 6

Let's run our docker container!

After running it, score this client once again:

```python
url = "YOUR_URL"
client = {"reports": 0, "share": 0.245, "expenditure": 3.438, "owner": "yes"}
requests.post(url, json=client).json()
```
What's the probability that this client will get a credit card now?

- 0.289
- 0.502
- 0.769
- 0.972
