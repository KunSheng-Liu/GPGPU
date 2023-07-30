# GPGPU

* Compile the code
    ```shell
    ./compile.sh
    ```

* Execute the code
    ```shell
    ./GPGPU --sm-num 8 --vram-pages 4096 -S Baseline  -T CaffeNet 5 0 -1 100  -T VGG16 2 0 -1 100  -T ResNet18 3 0 -1 100
    ```
    - -S
        > Baseline | Average | BARM | SALBI
    - -T [batch size] [arrival time] [period] [deadline]
        > LeNet | CaffeNet | ResNet18 | GoogleNet | VGG16


* Batch execution
    1. Prepare executable
    ```shell
    make RUN
    ```
    2. List the running command
    ```shell
    vim ./taskset_list.txt
    ```
    3. Run command in thread
    ```shell
    ./Run thread_num
    ```
