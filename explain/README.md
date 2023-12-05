# Overview
This folder contains an example where an application is integrated with responsible AI. In this example, tabular information representing a request for credit is provided to an AI model; the model makes a "Risk" or "No Risk" prediction for the account. The prediction could be used by a human to make informed decisions which will improve the business results of a fictitious bank.

![Application image](./container_image/application/static/slideshow/banking_app_1.jpg)

## Responsible AI Requirements
In order for this AI use case to be responsible, a few additional requirements have been added:
* __The process used to train, test, and deploy the model must be documented and repeatable__.
  * We must understand the training data, including any bias that might exist.
  * We must store metrics from our training and test process.
* __For accounts that are flagged as a risk, we need an explaination of why the model predicted "Risk"__.
* __Post deployment, the AI must be monitored__.
  * We must be able to determine when data drift has occurred.
  * we must report model performance metrics for production data.
* __The application must be "well engineered"__.
  * The inference service must be able to scale as user demand changes.
  * We need to be able to monitor the CPU & Memory usage of the application.

## Components
The requirements create a need for a number of components. The added complexity is what prevents many AI models from existing in production use cases. This demo shows how to build infrastructure to integrate the AI into the application.

![Demo components](./container_image/application/static/slideshow/banking_app_2.jpg)

## About the dataset
This example uses the German Credit Risk dataset, which can be obtained from the IBM Cloud [here](https://dataplatform.cloud.ibm.com/samples). You can create an IBM account for free to access the data.

The data is designed to be used as sample data for exploring IBM data science and AI tools. Models trained from this data are not suitable for real world applications, but the dataset is a wonderful resource for education and training use cases.

The data is distributed under an [MIT License](https://opensource.org/licenses/MIT).

# Installation Guide
This example has many components and is slightly more complicated to deploy than other examples.

The assumption is that the example will be deployed on:
* Red Hat OpenShift 4.12 running on IBM Power.
* Kubeflow 1.7 or newer
* IBM Power 9 or newer hardware

The container images in this example use a Python from RocketCE with is built with optimizations for IBM Power. These images only run on Power 9 or newer hardware.

The steps to deploy the application are outlined in this section.

## Install / Configure Database
The application and AI deploy/monitor pipelines both require an SQL database. The example includes multiple options for the database:
* PostgreSQL database running on OpenShift OR
* DB2 running on IBM Cloud

Follow the directions in README.md within the subdirectory for the database of your choice. You only need to setup one type of database.

## Build container images
If container images are not already built, you can build them by following the instructions in the `container_image` directory.

## Run Train/Eval/Deploy model pipeline
The `Deploy_Model_Pipeline.ipnb` notebook uploads and executes a pipeline to train, test, and deploy the model. The deployment includes a predictor, transformer, and explainer component.

You may need to make the following adjustments to the notebook:
1. The URLs for the images may need to be changed to the most recent versions of the images (espeically if new images have been build).
   
1. The credit_model_pipeline cell needs to be changed so that the two load data tasks use the correct component for the database.
   
   The two tasks are `load_training_data_task` and `load_test_data_task`
   
   Each task can choose to load from DB2 for PostgreSQL:
   * load_df_from_db2
   * load_df_from_postgresql
   <br/><br/>
   
   These tasks also need to be changed so that the correct method is called to add the right credentials for the database.
   For a DB2 connection, `add_db2_connection_secrets` should be called and `add_pg_connection_secrets` should be called for PostgreSQL.
   
Run the notebook to start the pipeline. The notebook flow will synchronize with the completion of the pipeline. If you click on the "Run Details" Link, you will be taken to the pipeline run. This graph serves as our documentation of the deployment process. Components of the graph can visualizations that document the characteristics of the training data, model performance and test results.

   ![Pipeline Run graph](./images/pipelinerun-graph.jpeg)
   
### Pipeline visualizations
In the graph of the pipeline has several important visualizations that can be viewed from visualizations pannel.
   * The Data Quality Report task includes a visualization for the quality of training data.
     This can be used to look for bias in the training data. For example, we can see that in our training data, the age of applicants classified as "Risk" is in general older than applicants classified as "No Risk".
     ![Age visualization](./images/pipeline-visualization-age.jpeg)
   * The Configure Tensorboard task includes a visualization with a link to a TensorBoard that monitors training in real time. The metrics displayed can be customized during training. The example shows how the loss and learning rate changes as the training progresses.
     ![TensorBoard](./images/tensorboard.jpeg)
   * The train component has a visualization that shows the precision-recall trade off for training data. This is important for binary classification problems because the graph describes model preformance across different thresholds.
     ![Precision Recall Curve](./images/pr-curve.jpeg)
   * The evaluate component has a visualization that shows metrics for how the model performed on the test data. Because the test data has not been used in the training, this gives an idea of how the model might preform on real world data. This report is generated using the Evidently AI package.
     ![Evaluation Chart](./images/evaluate-visualization.jpeg)
     
   
## Deploy Application
The deploy model pipeline deploys the inference service, but does not deploy the application. The application can be deployed from the command line, using one of the included yaml files.

* To deploy the model that depends on a DB2 database, use the `deploy_app_db2.yaml` file. 
* To deploy the model that depends on a PostgreSQL database, use the `deploy_app_postgresql.yaml` file.

You may need to make the following changes to the yaml file:
* The image may need to be changed to the most current image, expecially if you have rebuilt the application container image.
* The PREDICT_URL and EXPLAIN_URL environment variables may need to be updated to the namespace where the inference service is deployed.
* The annotations for autoscaling may need to be modified (if you want the application to dynamically scale based on cient demand).
  * autoscaling.knative.dev/max-scale
  * autoscaling.knative.dev/min-scale
  * autoscaling.knative.dev/target

You can deploy the application using the command `oc apply -f <filename>`.

Deploying the application creates a KNative route to access the application. You retrieve the URL for the application with this command:

`oc get ksvc demo-application -oyaml | yq -r '.status.url'`


## Create model monitoring recurring pipeline
In this step, we setup a recurring pipeline to detect data drift and report performance metrics for the AI model.

The pipeline is uploaded and executed using the `Deploy_Model_Monitoring.ipynb` notebook.

There are a few things in the notebook that may need to be changed:
* The base image needs to be set to the workflow image, this may need to be changed if you have rebuilt the container images.

* The monitor_credit_model_pipeline cell needs to be changed so that the tasks `load_reference_data_task` and `load_production_data_task` use the correct component. 

   Choose between the `load_df_from_db2` and `load_df_from_postgresql` components for DB2 and PostgreSQL respectively.
 
   These tasks also need to be changed so that the correct method is called to add the right credentials for the database.
   For a DB2 connection, `add_db2_connection_secrets` should be called and for PostgreSQL `add_pg_connection_secrets` should be called.
   
   
The notebook compiles and uploads the pipeline. After the upload, a recurring run is created.
   The recurring run happens every Monday-Friday at the top of the hour. The CRON syntax is used to define the interval and is described [here](https://pkg.go.dev/github.com/robfig/cron#hdr-CRON_Expression_Format).
   
You can delete recurring runs from the command line with the command:
   `oc delete scheduledworkflows --all`
   
# Using the application
   The application overview screen contains a slideshow with images that describe the implementation of the application and AI workflows. The images will include the correct database type for the configuration.
   
   ![Overview page](./images/overview.jpeg)
   
   There are buttons on the title bar to create and view accounts. The Risk Assessment is shown when viewing an accout. If the AI flags the account as a risk, a possible reason is included as explaination. This demo uses [Alibi Anchors](https://github.com/SeldonIO/alibi/blob/master/doc/source/methods/Anchors.ipynb) to explain the model's prediction.
   
   ![Risk Demo Image](./images/risk-demo.jpeg)
   
When viewing an account, the option to edit the account details is visible. The edit screen allows us to set the actual Risk or No Risk value for the account. Comparing the predicted risks vs the actual risks is a very important metric when assessing the correctness of the AI model. The account list screen displays the predicted risk and actual assigned risk side by side. Unknown values are common, because some amount of time will pass between the point of time the AI makes a prediction and the point of time when the actual value is known.

   ![List of accounts](./images/list-accounts.jpeg)

   
# Monitoring model performance and data drift
The monitoring pipeline runs at the specified times and produces reports and metrics. Pipeline runs are collected in the experiment `monitor-production-credit`.

The tasks of the pipeline runs contain visualizations with reports for:
* Data Drift
* Prediction quality
* Target value drift

The reports are only generated when a minimum number of accounts have changed within the time window (One week), in addition those accounts must have both actual and predicted values for risk. That allows us to look at the quality of that model over the time window in comparison to previous time windows.

The f1 score for the model is produced as a metric. These metrics can be quickly reviewed by look at the experiment, and help us understand how the model's real world performance changes over time. An experiment might look like this:

![experiment](./images/monitor-pipeline-runs.jpeg).

The reports are created using the Evidently AI package and can be viewed in the visualizations of the pipeline run:

![drift report](./images/drift-report.jpeg)

## Monitor the application using the OpenShift console
The "Observe" pannel of the OpenShift console allows the developer to view statistics about a service. For example we might wish to review the CPU usage of the predictor component.

![picture of CPU stats](./images/cpu-stats.jpeg)

### Monitor scalability
A service can be configured to autoscale as demands on the service changes by changing annotations in the specification of the KNative service.

Services can even be configured to scale to zero by setting min-scale equal to 0. This means that the service will cease to run when there has been recent demand for the service. The service will be started again when a request is made against the serivce.

If your OpenShift account has sufficient authorities, you can view a dashboard report that describes the auto scaling behavior of the KNative Service.

![auto scale report](./images/auto-scale.jpeg)

The notebook `create_lots_of_accounts_in_parallel.ipynb` has code in it to generate lots of new accounts in parallel. This notebook can be used to experiment with the autoscaling capabilities.