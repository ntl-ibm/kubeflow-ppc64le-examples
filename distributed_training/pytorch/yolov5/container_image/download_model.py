import logging
import boto3
from botocore import UNSIGNED
from botocore.client import Config
from urllib.parse import urlparse
import mimetypes
import os
import tarfile
import zipfile
import sys


def get_S3_config():
    # default s3 config
    c = Config()

    # anon environment variable defined in s3_secret.go
    anon = "true" == os.getenv("awsAnonymousCredential", "false").lower()
    # S3UseVirtualBucket environment variable defined in s3_secret.go
    # use virtual hosted-style URLs if enabled
    virtual = "true" == os.getenv("S3_USER_VIRTUAL_BUCKET", "false").lower()

    if anon:
        c = c.merge(Config(signature_version=UNSIGNED))
    if virtual:
        c = c.merge(Config(s3={"addressing_style": "virtual"}))

    return c


def download_s3(uri, temp_dir: str):
    # Boto3 looks at various configuration locations until it finds configuration values.
    # lookup order:
    # 1. Config object passed in as the config parameter when creating S3 resource
    #    if awsAnonymousCredential env var true, passed in via config
    # 2. Environment variables
    # 3. ~/.aws/config file
    kwargs = {"config": get_S3_config()}
    endpoint_url = os.getenv("AWS_ENDPOINT_URL")
    if endpoint_url:
        kwargs.update({"endpoint_url": endpoint_url})
    verify_ssl = os.getenv("S3_VERIFY_SSL")
    if verify_ssl:
        verify_ssl = not verify_ssl.lower() in ["0", "false"]
        kwargs.update({"verify": verify_ssl})
    else:
        verify_ssl = True

    # If verify_ssl is true, then check there is custom ca bundle cert
    if verify_ssl:
        global_ca_bundle_configmap = os.getenv("CA_BUNDLE_CONFIGMAP_NAME")
        if global_ca_bundle_configmap:
            isvc_aws_ca_bundle_path = os.getenv("AWS_CA_BUNDLE")
            if isvc_aws_ca_bundle_path and isvc_aws_ca_bundle_path != "":
                ca_bundle_full_path = isvc_aws_ca_bundle_path
            else:
                global_ca_bundle_volume_mount_path = os.getenv(
                    "CA_BUNDLE_VOLUME_MOUNT_POINT"
                )
                ca_bundle_full_path = (
                    global_ca_bundle_volume_mount_path + "/cabundle.crt"
                )
            if os.path.exists(ca_bundle_full_path):
                logging.info("ca bundle file(%s) exists." % (ca_bundle_full_path))
                kwargs.update({"verify": ca_bundle_full_path})
            else:
                raise RuntimeError(
                    "Failed to find ca bundle file(%s)." % ca_bundle_full_path
                )
    s3 = boto3.resource("s3", **kwargs)
    parsed = urlparse(uri, scheme="s3")
    bucket_name = parsed.netloc
    bucket_path = parsed.path.lstrip("/")

    file_count = 0
    exact_obj_found = False
    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=bucket_path):
        # Skip where boto3 lists the directory as an object
        if obj.key.endswith("/"):
            continue
        # In the case where bucket_path points to a single object, set the target key to bucket_path
        # Otherwise, remove the bucket_path prefix, strip any extra slashes, then prepend the target_dir
        # Example:
        # s3://test-bucket
        # Objects: /a/b/c/model.bin /a/model.bin /model.bin
        #
        # If 'uri' is set to "s3://test-bucket", then the downloader will
        # download all the objects listed above, re-creating their subpaths
        # under the temp_dir.
        # If 'uri' is set to "s3://test-bucket/a", then the downloader will
        # add to temp_dir: b/c/model.bin and model.bin.
        # If 'uri' is set to "s3://test-bucket/a/b/c/model.bin", then
        # the downloader will add to temp dir: model.bin
        # (without any subpaths).
        # If the bucket path is s3://test/models
        # Objects: churn, churn-pickle, churn-pickle-logs
        bucket_path_last_part = bucket_path.split("/")[-1]
        object_last_path = obj.key.split("/")[-1]
        bucket_path_parent_part = bucket_path.rsplit("/", 1)[0]

        if bucket_path == obj.key:
            target_key = obj.key.rsplit("/", 1)[-1]
            exact_obj_found = True
        elif object_last_path.startswith(bucket_path_last_part):
            target_key = obj.key.replace(bucket_path_parent_part, "", 1).lstrip("/")
        else:
            target_key = obj.key.replace(bucket_path, "").lstrip("/")

        target = f"{temp_dir}/{target_key}"
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target), exist_ok=True)
        bucket.download_file(obj.key, target)
        logging.info("Downloaded object %s to %s" % (obj.key, target))
        file_count += 1

        # If the exact object is found, then it is sufficient to download that and break the loop
        if exact_obj_found:
            break
    if file_count == 0:
        raise RuntimeError("Failed to fetch model. No model found in %s." % bucket_path)

    # Unpack compressed file, supports .tgz, tar.gz and zip file formats.
    if file_count == 1:
        mimetype, _ = mimetypes.guess_type(target)
        if mimetype in ["application/x-tar", "application/zip"]:
            _unpack_archive_file(target, mimetype, temp_dir)


def _unpack_archive_file(file_path, mimetype, target_dir=None):
    if not target_dir:
        target_dir = os.path.dirname(file_path)

    try:
        logging.info("Unpacking: %s", file_path)
        if mimetype == "application/x-tar":
            archive = tarfile.open(file_path, "r", encoding="utf-8")
        else:
            archive = zipfile.ZipFile(file_path, "r")
        archive.extractall(target_dir)
        archive.close()
    except (tarfile.TarError, zipfile.BadZipfile):
        raise RuntimeError(
            "Failed to unpack archive file. \
    The file format is not valid."
        )
    os.remove(file_path)


if __name__ == "__main__":
    download_s3(sys.args[1], sys.args[2])
