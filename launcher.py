import time
import subprocess
import os

all_n = [2, 3, 4, 5, 6, 7, 8]  # list of n_bits to test
num_gpus = 4                    # number of GPUs available
max_parallel = num_gpus
running = []

n_queue = all_n.copy()  # jobs waiting to run

while n_queue or any(r.poll() is None for r in running):
    # Launch jobs while GPUs are available
    for gpu_id in range(num_gpus):
        # Skip if this GPU is busy
        if any(r.poll() is None and r.gpu_id == gpu_id for r in running):
            continue
        if not n_queue:
            break

        n = n_queue.pop(0)
        print(f"Launching sweep for n_bits={n} on GPU {gpu_id}")

        folder = f"results/n{n}_sweep"
        os.makedirs(folder, exist_ok=True)

        cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python3.9 lenet5_quan_unsign_predict.py --n_bits {n}"
        log_path = os.path.join(folder, "log.txt")
        log_file = open(log_path, "w")

        p = subprocess.Popen(cmd, shell=True, stdout=log_file, stderr=log_file)
        p.gpu_id = gpu_id  # attach GPU info to process for tracking
        running.append(p)

    # Sleep a bit before checking again
    time.sleep(5)

# Wait for all remaining jobs to finish
for p in running:
    p.wait()

print("✅ All sweeps finished.")
