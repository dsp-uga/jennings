# Jennings
Celia segmentation


## Deploying to Google Cloud Dataproc
Dataproc is a managed YARN cluster service provided by Google. We provide a template script for easily provisioning a Dataproc cluster, submitting a Jennings job, and tearing down the cluster once the job completes.

To use the template script, make a copy of it and modify it for to your needs:

```shell
cp ./scripts/gcp-template.sh ./gcp.sh
vim ./gcp.sh
```

Take some time to read through the script. The script is primarily configured by a set of variables at the top. These all have sane, minimal defaults. The only variable that **must** be configured is the `BUCKET`, which should be set to a GCP bucket to which you have write access.

The script is designed to be launched from the root of this repository. To use a different working directory, update the relative paths accordingly.

Calling the script will submit a Jennings job to a newly created Dataproc cluster. The arguments passed to the script are forwarded to Jennings on the cluster.


## Running the tests
Jennings's test suite uses `pytest`.

**TODO:** More documentation
