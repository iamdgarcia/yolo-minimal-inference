import toml
import subprocess

def main():
    # Load the pyproject.toml file
    with open("pyproject.toml", "r") as f:
        pyproject = toml.load(f)

    # Extract the version
    version = pyproject["tool"]["poetry"]["version"]

    # Get the latest Git tag
    try:
        latest_tag = subprocess.check_output(["git", "describe", "--tags", "--abbrev=0"]).strip().decode()
    except subprocess.CalledProcessError:
        latest_tag = "v0.0.0"

    # Compare versions
    needs_release = version != latest_tag.lstrip("v")

    # Output results
    print(f"::set-output name=needs_release::{needs_release}")
    print(f"::set-output name=version::{version}")

if __name__ == "__main__":
    main()
