#!/usr/bin/env python3
import argparse
import toml
import sys

PYPROJECT_FILE = "pyproject.toml"

def parse_args():
    parser = argparse.ArgumentParser(description="Bump version in pyproject.toml")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--major", action="store_true", help="Bump major version")
    group.add_argument("--minor", action="store_true", help="Bump minor version")
    group.add_argument("--patch", action="store_true", help="Bump patch version")
    return parser.parse_args()

def read_version():
    try:
        data = toml.load(PYPROJECT_FILE)
        version_str = data["project"]["version"]
        return version_str, data
    except KeyError:
        print("Version konnte nicht gefunden werden. Stelle sicher, dass pyproject.toml '[project] version' enthält.")
        sys.exit(1)
    except FileNotFoundError:
        print(f"{PYPROJECT_FILE} nicht gefunden.")
        sys.exit(1)

def bump_version(version, major=False, minor=False, patch=False):
    parts = list(map(int, version.split(".")))
    while len(parts) < 3:  # fehlende Teile mit 0 auffüllen
        parts.append(0)
    
    if major:
        parts[0] += 1
        parts[1] = 0
        parts[2] = 0
    elif minor:
        parts[1] += 1
        parts[2] = 0
    elif patch:
        parts[2] += 1

    return ".".join(map(str, parts))

def write_version(data, new_version):
    data["project"]["version"] = new_version
    with open(PYPROJECT_FILE, "w") as f:
        toml.dump(data, f)
    print(f"Version erfolgreich auf {new_version} erhöht.")

def main():
    args = parse_args()
    version_str, data = read_version()
    new_version = bump_version(
        version_str,
        major=args.major,
        minor=args.minor,
        patch=args.patch
    )
    write_version(data, new_version)

if __name__ == "__main__":
    main()
