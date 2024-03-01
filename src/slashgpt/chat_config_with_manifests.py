import os
import re
import json
import boto3
from typing import Optional

import yaml

from slashgpt.chat_config import ChatConfig
from slashgpt.utils.print import print_error


class ChatConfigWithManifests(ChatConfig):
    """
    A subclass of ChatConfig, which maintains the set of manifests loaded from
    a specified folder.
    """

    def __init__(
        self,
        base_path: str,
        path_manifests: str,
        llm_models: Optional[dict] = None,
        llm_engine_configs: Optional[dict] = None,
    ):
        """
        Args:

            base_path (str): path to the "base" folder.
            path_manifests (str): path to the manifests folder (json or yaml)
            llm_models (dict, optional): collection of custom LLM model definitions
            llm_engine_configs (dict, optional): collection of custom LLM engine definitions
        """
        super().__init__(base_path, llm_models, llm_engine_configs)
        self.manifests: dict = self.__load_manifests(path_manifests)
        """Set of manifests loaded from the specified folder"""
        self.path_manifests: str = path_manifests
        """Location of the folder where manifests were loaded"""

    @classmethod
    def __load_manifests(cls, path: str):
        manifests = {}
        files = os.listdir(path)
        for file in files:
            if re.search(r"\.json$", file):
                with open(f"{path}/{file}", "r", encoding="utf-8") as f:  # encoding add for Win
                    try:
                        manifests[file.split(".")[0]] = json.load(f)
                    except json.JSONDecodeError:
                        print_error(file + " is broken")
            elif re.search(r"\.yml$", file):
                with open(f"{path}/{file}", "r", encoding="utf-8") as f:  # encoding add for Win
                    try:
                        manifests[file.split(".")[0]] = yaml.safe_load(f)
                    except Exception:
                        print_error(file + " is broken")
        return manifests

    def load_manifests_s3(self, bucket_name: str, prefix: str):
        s3 = boto3.client("s3")
        manifests = {}

        # List objects within the specified bucket and prefix
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        if "Contents" in response:
            for obj in response["Contents"]:
                file_name = obj["Key"]
                # TNE naming convention - manifests have .model extension
                if file_name.endswith(".json") or file_name.endswith(".yml") or file_name.endswith(".yaml") or file_name.endswith(".model"):
                    # Get the object from S3
                    file_obj = s3.get_object(Bucket=bucket_name, Key=file_name)
                    # Read the file content
                    file_content = file_obj["Body"].read().decode("utf-8")

                    # Determine the file type and load accordingly
                    if file_name.endswith(".json"):
                        manifests[file_name.split("/")[-1].split(".")[0]] = json.loads(file_content)
                    elif file_name.endswith(".yml") or file_name.endswith(".yaml") or file_name.endswith(".model"):
                        manifests[file_name.split("/")[-1].split(".")[0]] = yaml.safe_load(file_content)

        return manifests

    def load_manifests_s3(self, bucket_name: str, prefix: str):
        s3 = boto3.client("s3")
        manifests = {}

        # List objects within the specified bucket and prefix
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        if "Contents" in response:
            for obj in response["Contents"]:
                file_name = obj["Key"]
                # TNE naming convention - manifests have .model extension
                if file_name.endswith(".json") or file_name.endswith(".yml") or file_name.endswith(".yaml") or file_name.endswith(".model"):
                    # Get the object from S3
                    file_obj = s3.get_object(Bucket=bucket_name, Key=file_name)
                    # Read the file content
                    file_content = file_obj["Body"].read().decode("utf-8")

                    # Determine the file type and load accordingly
                    if file_name.endswith(".json"):
                        manifests[file_name.split("/")[-1].split(".")[0]] = json.loads(file_content)
                    elif file_name.endswith(".yml") or file_name.endswith(".yaml") or file_name.endswith(".model"):
                        manifests[file_name.split("/")[-1].split(".")[0]] = yaml.safe_load(file_content)

        return manifests

    def switch_manifests(self, path: str):
        """Switch the set of manifests

        Args:

            path (str): path to the manifests folder (json or yaml)
        """
        self.path_manifests = path
        self.reload()

    def reload(self):
        """Reload manifest files"""
        self.manifests = self.__load_manifests(self.path_manifests)

    def has_manifest(self, key: str):
        """Check if a manifest file with a specified name exits
        Args:

            key (str): the name of manifest
        """
        return key in self.manifests
