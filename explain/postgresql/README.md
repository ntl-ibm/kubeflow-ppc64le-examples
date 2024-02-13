# Setup a PostgreSQL database
The demo application can be setup to use a PostgreSQL database running in the same namespace as the application and Kubeflow Pipelines.

## OpenShift templates
OpenShift includes templates for deploying a PostgreSQL database. This makes it easier to deploy.

Because Kubeflow uses Istio to manage network traffic and PostgreSQL is not designed to use Istio injection, we had to modify the default template to disable Istio injection.

The modified template is in `postgresql-no-istio-persistent.yaml`.

You can use the `oc` command to process the template and deploy PostgreSQL. 
The terminal in a Jupyter Notebook does not have sufficient authority to process a template, the workaround is:

1) Open a terminal to the bastion node (The node with the oc command)
2) Log into OpenShift.  (You can get a login token from the OpenShift console. Choose "Copy Login command from the drop down under your email in the top right corner).
3) Clone this github repo. git clone https://github.com/ntl-ibm/kubeflow-ppc64le-examples.git -b 3.0.0
4) cd to this working directory
5) Run the following command

`oc process  -f ./postgresql-no-istio-persistent.yaml   -l component=credit-risk-pg -p POSTGRESQL_USER=pguser -p POSTGRESQL_DATABASE=credit-risk | oc create -f -`

## Create Schema
Once the database has been deployed, run the `SETUP-For-PostreSQL.ipynb` notebook to create the schema.

The notebook creates the application table, as well as tables for training and testing the model. 

It also creates the (unrelated) credentials used by InferenceService to access MinIO.