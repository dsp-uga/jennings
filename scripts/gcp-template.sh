#!/bin/bash

# Configuration
# --------------------------------------------------
# The default configuration should be minimal when checking into git.
# That means no workers, no GPUs, and a n1-standard-1 machine type.
#
# Notes on GPU instances:
# - You must uncomment the accelerator args in the "Provision" section.
# - You must use at least 4 cores. The init process requires a compile
#   stage that will timeout if not enough CPUs are available.

BUCKET="MY_BUCKET"           # The bucket to stage scripts, recieve cluster output
CLUSTER="jennings-${RANDOM}"  # Name of the cluster to create
REGION="us-east1"            # Region. Not all regions have GPUs
ZONE="us-east1-d"            # Zone. Not all zones have GPUs: https://cloud.google.com/compute/docs/gpus/
WORKERS="0"                  # Number of workers in the cluster
TYPE="n1-standard-1"         # Machine type for master and workers
GPUS="0"                     # Number of GPUs per machine
GPU_TYPE="nvidia-tesla-k80"  # GPU type


# Logging
# --------------------------------------------------
echo "==> INFO"
echo "command: $@"
echo "--> Google Cloud"
echo "cluster: $CLUSTER"
echo "region: $REGION"
echo "zone: $ZONE"
echo "workers: $WORKERS"
echo "machine type: $TYPE"
echo "GPUs: $GPUS"
echo "GPU Type: $GPU_TYPE"
echo


# Compile
# --------------------------------------------------
echo "==> Compiling the module"
./setup.py bdist_egg
echo


# Upload scripts
# --------------------------------------------------
echo "==> Uploading scripts"
gsutil rsync "./scripts" "gs://$BUCKET/jennings/scripts"
echo


# Provision
# --------------------------------------------------
# Uncomment the accellerators to enable GPUs
echo "==> Provisioning cluster $CLUSTER in $ZONE ($REGION)"
gcloud beta dataproc clusters create "$CLUSTER" \
	--region "$REGION" \
	--zone "$ZONE" \
	--bucket "$BUCKET" \
	--master-machine-type "$TYPE" \
	--worker-machine-type "$TYPE" \
	--num-workers "$WORKERS" \
	--initialization-action-timeout "20m" \
	--initialization-actions "gs://$BUCKET/jennings/scripts/gcp-bootstrap.sh" \
	# --master-accelerator "type=$GPU_TYPE,count=$GPUS" \
	# --worker-accelerator "type=$GPU_TYPE,count=$GPUS" \

echo


# Action
# --------------------------------------------------
# The ssh action is useful for developing and debugging.
# Leave the ssh action commented out when checking this file into git.

# echo "==> Connecting to $CLUSTER-m via SSH:"
# gcloud compute ssh "$CLUSTER-m" \
# 	--zone "$ZONE"
# echo

echo "==> Submitting job: $@"
gcloud dataproc jobs submit pyspark \
	--cluster "$CLUSTER" \
	--region "$REGION" \
	--driver-log-levels root=FATAL \
	--py-files ./dist/jennings-*.egg \
	./scripts/driver.py \
	-- $@
echo


# Teardown
# --------------------------------------------------
echo "==> Tearing down the cluster"
gcloud dataproc clusters delete "$CLUSTER" \
	--region "$REGION" \
	--quiet
echo
