# Instructions for building the image

There are many container registries. We use quay.io. If you would like to also use quay.io, you will need to create a free account First.

## Create a quay.io account. (If necessary)

1.  In a web browser, navigate to quay.io and click sign in
1.  You will be asked to Log into your Red Hat account
1.  Choose Register for a Red Hat Account

## Build your container image on an IBM Power server and push it to quay

1. Copy the Docker file to a directory on your IBM Power server (one that has podman installed)
1. Build the container image locally, and tag it with your remote repository and tag
   `podman build -f Dockerfile --format docker -t quay.io/<your-user-id-here>/spacy-nb:v1.0.1`
1. Login to quay.io
   `podman login`
   Provide your credentials when asked.
1. Push the image to your repository.
   `podman push quay.io/<your-user-id-here>/spacy-nb:v1.0.1`

   We suggest increasing the version number each time you change your image, this will avoid accidently using the wrong version.

## Make your repository public

In order to use (pull) your image in a K8S cluster, you must first make the image public. The alternative is to provide k8s and kubeflow with a "Pull Secret", but this is generally too complicated for non-sensitive container images.

1.  In a web browser, go to quay.io and sign in with your credentials.
1.  Find the repository in the list and click on it.
1.  On the left hand pane, click settings (The gear icon)
1.  Under "Repository Visibility" click "Make Visible"

You can now use the container image as a base image for your notebook server or kubeflow pipeline component!
