# Generate template file
# oc get template  postgresql-persistent -n openshift  -oyaml > postgresql-no-istio-persistent.yaml
# remove opneshift namespace from the template
# add:  sidecar.istio.io/inject: "false" to DeploymentConfig.spec.template.metadata
oc process  -f ./postgresql-no-istio-persistent.yaml   -l component=credit-risk-pg -p POSTGRESQL_USER=ntl -p POSTGRESQL_DATABASE=credit-risk | oc create -f -
