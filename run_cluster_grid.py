#! /usr/bin/env python3

import os
import subprocess
import time


# training teachers for now


def main():
    username = os.environ["USER"]

    processes = []
    commands = []
    name2training = {
        "celeba": "celeba",
        "cifar": "cifar",
        "power": "tabular",
        "gas": "tabular",
        "hepmass": "tabular",
        "miniboone": "tabular",
        "bsds300": "tabular",
    }
    for name in ["celeba", "cifar", "power", "gas", "hepmass", "miniboone", "bsds300"]:
        for _ in range(3):
            commands.append(
                f"sbatch -t 15000 --gpus=1 -p normal -c 4 run_cluster.sh "
                f"dataset={name} teacher={name} student={name} "
                f"training={name2training[name]}"
            )

    batch_size = 20
    for command in commands:
        print(command)
        process = subprocess.Popen(
            command,
            shell=True,
            close_fds=True,
        )
        processes.append(process)
        pr_count = subprocess.Popen(
            f"squeue | grep {username} | wc -l", shell=True, stdout=subprocess.PIPE
        )
        out, err = pr_count.communicate()
        if int(out) > batch_size:
            while int(out) > batch_size:
                print("Waiting... ")
                time.sleep(240)
                pr_count = subprocess.Popen(
                    f"squeue | grep {username} | wc -l",
                    shell=True,
                    stdout=subprocess.PIPE,
                )
                out, err = pr_count.communicate()

    for process in processes:
        print(process.pid)


if __name__ == "__main__":
    main()
