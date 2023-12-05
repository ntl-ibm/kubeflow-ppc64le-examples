# Setup a PostgreSQL database
The demo application can be setup to use a PostgreSQL database running in the same namespace as the application and Kubeflow Pipelines.

## OpenShift templates
OpenShift contains templates for deploying a PostgreSQL database. This makes it easier to deploy.

Because Kubeflow uses Istio to manage network traffic and PostgreSQL is not designed to use Istio, we had to modify the default template to disable Istio injection.

The modified template is in `postgresql-no-istio-persistent.yaml`.

Run the following oc command to create the results in the template. You may need to run this from a command line, rather than a terminal in Jupyter.

`oc process  -f ./postgresql-no-istio-persistent.yaml   -l component=credit-risk-pg -p POSTGRESQL_USER=ntl -p POSTGRESQL_DATABASE=credit-risk | oc create -f -`

## Create Schema
Once the database has been deployed, run the `SETUP-For-PostreSQL.ipynb` notebook to create the schema.

This creates the application table, as well as tables for training and testing the model. It also creates the credentials used by InferenceService to access MinIO.