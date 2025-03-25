export prog_root=$(dirname $(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd))

# Customize these variables according to your directory structure
export storage_root="${HOME}/projects/storage"
export output_parent_path="$storage_root/outputs/empirical_peft_suite"
