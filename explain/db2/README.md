# Setup a DB2 on Cloud Database
This document describes the steps for setting up a DB2 on Cloud Database and creating the schema for this example.

## Log into the IBM Cloud.
The IBM cloud may be accessed [here](https://cloud.ibm.com)
You can create an account for free by clicking "Create and Accout" if you do not have an IBMId.

## Deploy a DB2 Database
You can deploy a trial version of a DB2 Database for Free.

* Open the resource list using the sidebar on the left.
* Click the "Create resource" button on the top right.
* Type "Db2" into the search window
* Choose the "Lite" Plan
* Accept the terms on right hand side
* Click "Create"

The database may take a few minutes to deploy.

## Obtain connection details
After the database becomes active, you can obtain the connection details.

* Find the database in the resource list and click on it.
* Choose "Service credentials" from the left bar
* Expand the "admin" credential
* From the "connection" -> "cli" -> "arguments" collect the following information:
  * username (-u argument)
  * password (-p argument)
  * host and port (--host option, the port is after the ":" on the end, the host is before the ":")
  
## Set the connection details into the secret template
In this directory, edit `db2-secret.yaml` to set the username, password, host, and port from the previous step.

## Create the Secret, and DB2 schema.
Open the `SETUP-For-DB2.ipynb` notebook in this example and run all cells.

This will create the schema for the application, as well as training and test data to deploy the model.

This script will also create the secret that provides the credentials to MinIO to the inference service, this is unrelated to DB2, but is required to deploy the inference service.



