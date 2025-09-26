## Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# activate venv
uv venv --clear
source .venv/bin/activate
uv sync && uv pip install -e .
uv pip install boto3

sudo mkdir ~/.kube &&\
sudo mv /scripts/kube/config ~/.kube/config

export AWS_PROFILE=greenland
export KUBECTL_VERSION=1.30.2
curl -LO "https://dl.k8s.io/release/v${KUBECTL_VERSION}/bin/linux/amd64/kubectl" && \
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl && \
rm kubectl
kubectl cluster-info
kubectl delete --all pods --namespace=default-zxugt
echo ">>>>>>>>>>>>>>>>>>"

# Start background S3 sync process
sync_traj_to_s3() {
    while true; do
        if [ -d "./traj" ]; then
            aws s3 sync ./traj s3://shopqa-users/zxugt/results/traj --quiet
        fi
        sleep 30
    done
}

# Start the sync process in background
sync_traj_to_s3 &
SYNC_PID=$!

# Trap to cleanup background process on script exit
cleanup() {
    echo "Stopping background sync process..."
    kill $SYNC_PID 2>/dev/null
    # Final sync before exit
    if [ -d "./traj" ]; then
        aws s3 sync ./traj s3://shopqa-users/zxugt/results/traj
    fi
}
trap cleanup EXIT

echo "Started background S3 sync for ./traj (PID: $SYNC_PID)"

# Run the main training process
uv run python src/r2egym/agenthub/run/edit.py runagent_multiple \
  --traj_dir "./traj" \
  --max_workers 8 \
  --start_idx 4 \
  --k 46 \
  --dataset "R2E-Gym/R2E-Gym-Lite" \
  --split "train" \
  --llm_name 'bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0' \
  --use_fn_calling True \
  --exp_name r2egym-training-trajectories \
  --temperature 0.2 \
  --max_steps 40