#!bin/bash
git fetch origin && git reset --hard origin/main
cmake --build build/Debug
sbatch jobscript
