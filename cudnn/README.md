## Building and running the Code

Run the following to compile and run the code:

```bash
cd cudnn
docker-compose up -d # start the cotaniner
docker-compose exec cuda_dev bash # enter the container
./compile_benchmark.sh # or ./compile_toynetwork.sh
./runner
# Sample output for benchmark:
Average forward pass time: 0.185160 ms
Average backward input pass time: 0.161860 ms
Average backward params pass time: 0.030010 ms
Total time: 0.377030 ms
```