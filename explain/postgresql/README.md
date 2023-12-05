# Setup a PostgreSQL database
The demo application can be setup to use a PostgreSQL database running in the same namespace as the application and Kubeflow Pipelines.

## OpenShift templates
OpenShift includes templates for deploying a PostgreSQL database. This makes it easier to deploy.

Because Kubeflow uses Istio to manage network traffic and PostgreSQL is not designed to use Istio injection, we had to modify the default template to disable Istio injection.

The modified template is in `postgresql-no-istio-persistent.yaml`.

Run the following oc command to create the results in the template. You may need to run this from a command line, rather than a Jupyter terminal.

`oc process  -f ./postgresql-no-istio-persistent.yaml   -l component=credit-risk-pg -p POSTGRESQL_USER=pguser -p POSTGRESQL_DATABASE=credit-risk | oc create -f -`

## Create Schema
Once the database has been deployed, run the `SETUP-For-PostreSQL.ipynb` notebook to create the schema.

The notebook creates the application table, as well as tables for training and testing the model. 

It also creates the (unrelated) credentials used by InferenceService to access MinIO.