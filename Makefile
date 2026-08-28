.PHONY: run_vm run_local prepare_data test

# Override per run:  make run_vm IMG_SIZE=256
# Falls back to .env, then to 128 (see app/params.py).
IMG_SIZE ?= 128

# The only way to train. Trains, archives the run to the GCS bucket, evaluates.
#
# There is deliberately no "train without saving" target: models/current/ is
# overwritten by every run, so a training run that is not archived by
# save_model() is lost the moment the next one starts.
run_vm:
	IMG_SIZE=$(IMG_SIZE) MACHINE=vm python run_training.py

# Same training run on a laptop. Archived as classifier_local_... so a run
# trained off the VM is never mistaken for one that came from it.
run_local:
	IMG_SIZE=$(IMG_SIZE) MACHINE=local python run_training.py

# Build the splits and print their composition, without training.
prepare_data:
	IMG_SIZE=$(IMG_SIZE) python -c "from app.classification.main import prepare_data; prepare_data()"

# Smoke tests -- run before pushing. Same suite GitHub Actions runs on PRs.
test:
	pytest tests/ -v
