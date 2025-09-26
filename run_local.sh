mkdir ~/.kube &&\
cp /scripts/kube/config ~/.kube/config

## Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# activate venv
uv venv
source .venv/bin/activate
uv sync && uv pip install -e .
uv pip install boto3

export AWS_PROFILE=default
export KUBECTL_VERSION=1.30.2
curl -LO "https://dl.k8s.io/release/v${KUBECTL_VERSION}/bin/linux/amd64/kubectl" && \
install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl && \
rm kubectl
kubectl cluster-info
uv run python src/r2egym/agenthub/run/edit.py runagent_multiple \
  --traj_dir "./traj" \
  --max_workers 4 \
  --start_idx 0 \
  --k 2 \
  --dataset "R2E-Gym/R2E-Gym-Lite" \
  --split "train" \
  --llm_name 'bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0' \
  --use_fn_calling True \
  --exp_name r2egym-training-trajectories \
  --temperature 0.2 \
  --max_steps 40
aws s3 cp ./traj s3://shopqa-users/zxugt/results/traj --recursive

# curl -LsSf https://astral.sh/uv/install.sh | sh
# source $HOME/.local/bin/env

# # # activate venv
# uv venv
# source .venv/bin/activate
# uv sync && uv pip install -e .
# uv pip install boto3

# # aws eks update-kubeconfig --region us-east-2 --name second-trial --profile $AWS_PROFILE
# # exit
# kubectl cluster-info
# kubectl delete --all pods --namespace=default-zxugt
# echo ">>>>>>>>>>>>>>>>>>"
# # export AWS_PROFILE=ShopQA-alpha && \
# uv run python src/r2egym/agenthub/run/edit.py runagent_multiple \
#   --traj_dir "./traj" \
#   --max_workers 8 \
#   --start_idx 4 \
#   --k 50 \
#   --dataset "R2E-Gym/R2E-Gym-Lite" \
#   --split "train" \
#   --llm_name 'bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0' \
#   --use_fn_calling True \
#   --exp_name r2egym-training-trajectories \
#   --temperature 0.2 \
#   --max_steps 40
# aws s3 cp ./traj s3://shopqa-users/zxugt/results/traj --recursive