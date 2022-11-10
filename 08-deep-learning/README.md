# Neural Networks and Deep Learning

## 8.1 Fashion Classification

In this session, we'll be working with multiclass image classification with deep learning. The deep learning frameworks like TensorFlow and Keras will be implemented on clothing dataset to classify images of t-shirts.

The dataset has 5000 images of 20 different classes, however, we'll be using the subset which contains 10 of the most popular classes. Following is the link to download the dataset:

```bash
git clone https://github.com/alexeygrigorev/clothing-dataset-small.git
```

**Userful links**:

- Full dataset: https://www.kaggle.com/agrigorev/clothing-dataset-full
- Subset: https://github.com/alexeygrigorev/clothing-dataset-small
- Corresponding Medium article: https://medium.com/data-science-insider/clothing-dataset-5b72cd7c3f1f
- CS231n CNN for Visual Recognition: https://cs231n.github.io/

## 8.1b Setting Up the Environment on Saturn Cloud

Following are the instructions to create an SSH private and public key and setup Saturn Cloud for GitHub:

1. [Generating a new SSH key and adding it to the ssh-agent](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
2. [Adding a new SSH key to your GitHub account](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account?tool=webui)
3. Then we just need to follow the video (8.1b) of session 8 to add the ssh keys to secrets and authenticate through a terminal

Alternatively, we could just use the public keys provided by Saturn Cloud by default. To do so, we need to follow these steps:

- From Saturn Cloud dashboard, click on the username and then on manage
- Down blow we will see the Git SSH Keys section, copy the provided default public key
- Paste these keys into the SSH keys section of our github repo
- Open the terminal on Saturn Cloud and run the command `ssh -T git@github.com`
- We should receive a successful authentication notice

## 8.2 TensorFlow and Keras

