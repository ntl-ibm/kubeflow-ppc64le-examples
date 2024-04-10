# Setup a PostgreSQL database

The demo application can be setup to use a PostgreSQL database running in the same namespace as the application and Kubeflow Pipelines.

**_The `oc` commands used in this readme MUST be run from a command line from a terminal on SCOUT. A notebook terminal does not have authority to work with templates._**

## OpenShift templates

OpenShift includes templates for deploying a PostgreSQL database. This makes it easier to deploy.
You can get the list of available templates using:
`oc get templates -n openshift`

The unmodified template for postgreSQL can be retrieved with this command:
`oc get template  postgresql-persistent -n openshift  -oyaml > postgresql-no-istio-persistent.yaml`.

Because Kubeflow uses Istio to manage network traffic and PostgreSQL is not designed to use Istio injection, we had to modify the default template to disable Istio injection. We did this by adding an annotation `sidecar.istio.io/inject: "false"` to the template.

The modified template is in `postgresql-no-istio-persistent.yaml`.

You can use the `oc` command to process the template and deploy PostgreSQL.

1. Open a terminal to the bastion node (The node with the oc command)
2. Log into OpenShift. (You can get a login token from the OpenShift console. Choose "Copy Login command from the drop down under your email in the top right corner).
3. Clone this github repo. git clone https://github.com/ntl-ibm/kubeflow-ppc64le-examples.git -b 3.0.0
4. cd to this working directory (kubeflow-ppc64le-examples/explain/postgresql)
5. Verify your current project is set correctly: `oc project <namespace>`
6. Run the following command

`oc process  -f ./postgresql-no-istio-persistent.yaml   -l component=credit-risk-pg -p POSTGRESQL_USER=pguser -p POSTGRESQL_DATABASE=credit-risk | oc create -f -`

## Create Schema

Once the database has been deployed, run the `SETUP-For-PostreSQL.ipynb` notebook to create the schema.

The notebook creates the application table, as well as tables for training and testing the model.

It also creates the (unrelated) credentials used by InferenceService to access MinIO.
