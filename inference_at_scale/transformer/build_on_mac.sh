#!/bin/bash
# If you want to do a docker build on a power9 or power10 machine, then you don't need this script. 
# Just do a podman build --format=docker -f Dockerfile -t quay.io/ntlawrence/monkeytransform:${VERSION} -t quay.io/ntlawrence/monkeytransform:latest
# Then do the podman push commands
# The podman buildx --platform  version of the command is not needed.
#
# But if you want to build a container that runs on IBM Power from a MAC M1, then this is the script for you.
# On a MAC M1, you'll need to do a few things to get podman to work (https://github.com/containers/podman/discussions/12899)
# * brew install podman 
# * podman machine init
# * podman machine start
# * podman machine ssh "bash -c 'sudo rpm-ostree install qemu-user-static && sudo systemctl reboot'"
#
# You'll need to do a podman machine start after each reboot
#
set -ex
VERSION="v4.1"
podman buildx build --format=docker --platform linux/ppc64le -f Dockerfile -t quay.io/ntlawrence/monkeytransform:${VERSION} -t quay.io/ntlawrence/monkeytransform:latest
podman push quay.io/ntlawrence/monkeytransform:latest quay.io/ntlawrence/monkeytransform:${VERSION}
podman push quay.io/ntlawrence/monkeytransform:latest quay.io/ntlawrence/monkeytransform:latest
