.PHONY: run_vm train prepare_data

# Override per run:  make run_vm IMG_SIZE=256
# Falls back to .env, then to 128 (see app/params.py).
IMG_SIZE ?= 128

# Train on the VM, push the model to the GCS bucket, then evaluate.
run_vm:
	IMG_SIZE=$(IMG_SIZE) python run_vm.py

# Train only, no bucket upload (local checkpoint still written).
train:
	IMG_SIZE=$(IMG_SIZE) python -m app.classification.train

# Build the splits and print their composition, without training.
prepare_data:
	IMG_SIZE=$(IMG_SIZE) python -c "from app.classification.main import prepare_data; prepare_data()"
