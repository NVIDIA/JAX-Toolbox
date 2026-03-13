import json
import os
import sys
from typing import Dict, List


def getenv(name: str) -> str:
    """Function to get env variables from action

    Args:
        name: name of the env variable to get

    Returns:
        value of the env variable
    """
    value = os.environ.get(name)
    if value is None:
        raise KeyError(f"Missing required environment variable: {name}")
    return value


def non_empty_lines(value: str) -> List[str]:
    """Function to clean the input from env variables

    Args:
        value: string to clean
    Returns:
        list of non empty lines in the input string
    """
    return [line for line in value.splitlines() if line != ""]


def kv_lines_to_object(value: str) -> Dict[str, str]:
    """Function to convert key value lines to a dict

    Args:
        value: string to convert, expected format is key=value per line
    Returns:
        dict with key value pairs from the input string
    """
    result: Dict[str, str] = {}
    for line in non_empty_lines(value):
        key, sep, rest = line.partition("=")
        result[key] = rest if sep else ""
    return result


def normalize_dockerfile_path(docker_context: str, dockerfile: str) -> str:
    """Function to normalize the dockerfile path to be relative to the docker context

    Args:
        docker_context: the docker context path
        dockerfile: the dockerfile path, can be absolute or relative to the context
    Returns:
        the dockerfile path relative to the docker context"""
    context_prefix = f"{docker_context.rstrip('/')}/"
    if dockerfile.startswith(context_prefix):
        return dockerfile[len(context_prefix) :]
    return dockerfile


def main() -> int:
    """Main function to generate the bake definition for the build container action
    Here we are creating the json file for docker/bake.
    The bake definition will have 3 targets:
    - mealkit: this target will build the mealkit image and push it to the
    registry with the specified tags and labels
    - final: this target will build the final image and push it to the registry
    with the specified tags and labels
    - cache-export: this target exports Bazel cache directories to local output
    """
    try:
        mealkit_tags = non_empty_lines(getenv("MEALKIT_TAGS"))
        final_tags = non_empty_lines(getenv("FINAL_TAGS"))
        mealkit_labels = kv_lines_to_object(getenv("MEALKIT_LABELS"))
        final_labels = kv_lines_to_object(getenv("FINAL_LABELS"))

        docker_context = getenv("DOCKER_CONTEXT")
        dockerfile = normalize_dockerfile_path(docker_context, getenv("DOCKERFILE"))
        architecture = getenv("ARCHITECTURE")
        base_image = getenv("BASE_IMAGE")
        bazel_cache = getenv("BAZEL_CACHE")
        build_date = getenv("BUILD_DATE")
        extra_build_args = kv_lines_to_object(getenv("EXTRA_BUILD_ARGS"))
        ssh_known_hosts_file = getenv("SSH_KNOWN_HOSTS_FILE")
        bazel_repo_context = getenv("BAZEL_REPO_CONTEXT")
        bazel_disk_context = getenv("BAZEL_DISK_CONTEXT")
        bazel_export_dir = getenv("BAZEL_EXPORT_DIR")
        cache_repo = getenv("BUILDKIT_CACHE_REPO")
        cache_key_prefix = getenv("BUILDKIT_CACHE_KEY_PREFIX")
        cache_branch = getenv("BUILDKIT_CACHE_BRANCH")
        bake_file = getenv("BAKE_FILE")
        branch_cache_ref = f"{cache_repo}:{cache_key_prefix}-{cache_branch}"
        main_cache_ref = f"{cache_repo}:{cache_key_prefix}-main"

        build_args: Dict[str, str] = {
            "BASE_IMAGE": base_image,
            "BUILD_DATE": build_date,
        }
        if bazel_cache:
            build_args["BAZEL_CACHE"] = bazel_cache
        build_args.update(extra_build_args)

        common = {
            "context": docker_context,
            "dockerfile": dockerfile,
            "platforms": [f"linux/{architecture}"],
            "contexts": {
                "bazel_repo": bazel_repo_context,
                "bazel_disk": bazel_disk_context,
            },
            "ssh": ["default"],
            "secret": [f"id=SSH_KNOWN_HOSTS,src={ssh_known_hosts_file}"],
            "args": build_args,
            "cache-from": [
                f"type=registry,ref={branch_cache_ref},ignore-error=true",
                f"type=registry,ref={main_cache_ref},ignore-error=true",
            ],
        }
        # create the json file for the bake action
        bake_definition = {
            "target": {
                "mealkit": {
                    **common,
                    "target": "mealkit",
                    "tags": mealkit_tags,
                    "labels": mealkit_labels,
                    "output": ["type=image,push=true"],
                },
                "final": {
                    **common,
                    "target": "final",
                    "tags": final_tags,
                    "labels": final_labels,
                    "output": ["type=image,push=true"],
                    "cache-to": [
                        "type=registry,"
                        f"ref={branch_cache_ref},"
                        "mode=max,oci-mediatypes=true,image-manifest=true,"
                        "compression=zstd,ignore-error=true"
                    ],
                },
                "cache-export": {
                    **common,
                    "target": "cache-export",
                    "output": [f"type=local,dest={bazel_export_dir}"],
                },
            }
        }

        with open(bake_file, "w", encoding="utf-8") as f:
            json.dump(bake_definition, f, separators=(",", ":"))
            f.write("\n")
        return 0
    except Exception as e:
        print(f"Failed to generate bake definition: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
