# prepare_avon_bundle.sh
#!/usr/bin/env bash
set -euo pipefail

# create standard places
mkdir -p scripts env configs

# write (or overwrite) rsync exclude list
cat > rsync_exclude.txt <<'EOF'
.git
.venv
__pycache__
*.pyc
logs
data
.DS_Store
*.ipynb_checkpoints
EOF

echo "Created scripts/, env/, configs/ and rsync_exclude.txt"
