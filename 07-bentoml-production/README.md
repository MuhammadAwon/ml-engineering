# 7. BentoML Production

## 7.1 Intro and Overview

The goal of this session is to build and deploy an ML service, customize our service to fit our use case, and make our service production ready with the open-source library BentoML. For this, we'll be using the model we built in session 6.

What is production ready?

- `Scalability`: it is the ability to increase or decrease the resources of the application according to the user demands.
- `Operationally efficiency`: it is being able to maintain the service by reducing the time, efforts and resources as much as possible without compromising the high-quality.
- `Repeatability (CI/CD)`: to update or modify the service as we need without having to do everything again from the scratch.
- `Fexibility`: to make it easy to react and apply changes to the issues in the production.
- `Resiliency`: to ensure even if the service is completely broke we are still able to reverse to the previous stable version.
- `Easy to use`: all the required frameworks should be easy to use.

We first focus should always be on getting the service to the production and rest will come later.

What is a BentoML? A typical machine learning application has various components. For instance, code, model(s), data, dependencies, configuration, deployment logic, and many more. BentoML packages all these componments into one deployable unit with ease.

What we will be convering in session 7?

- Building a prediction service
- Deploying our prediction service
- Anatomy of a BentoML service
- Customizing Bentos
- BentoML Production Deployment
- High Perfromance serving
- Custom Runner / Framework

## 7.2 Build Bento Service

## 7.3 