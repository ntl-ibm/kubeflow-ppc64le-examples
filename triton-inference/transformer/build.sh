#!/bin/bash
# On a MAC, you'll need to do a few things to get podman to work (https://github.com/containers/podman/discussions/12899)
# * brew install podman 
# * podman machine init
# * podman machine start
# * podman machine ssh "bash -c 'sudo rpm-ostree install qemu-user-static && sudo systemctl reboot'"
#
# You'll need to do a podman machine start after each reboot
#
VERSION="v2.0"
podman buildx build --format=docker --platform linux/ppc64le -f Dockerfile -t quay.io/ntlawrence/monkeytransform:${VERSION} -t quay.io/ntlawrence/monkeytransform:latest
podman push quay.io/ntlawrence/monkeytransform:latest quay.io/ntlawrence/monkeytransform:${VERSION}
