#!/bin/bash

# Define the remote server and directory
REMOTE_SERVER="fcruzfal@login.hpc.caltech.edu"
REMOTE_DIR="/groups/CS156b/data/train"

# Define the local directory where the tarball will be downloaded
LOCAL_DIR="/home/pipec/cs156b/"

# Define the name of the tarball
TARBALL_NAME="cs156_train_data_small.tar.gz"

# Create a tarball of the first 100 files in the remote directory and download it
ssh $REMOTE_SERVER "cd $REMOTE_DIR && ls | head -n 500 | tar -czf - -T -" | cat > $LOCAL_DIR/$TARBALL_NAME

# Print a message to indicate the download is complete
echo "Download complete: $LOCAL_DIR/$TARBALL_NAME"