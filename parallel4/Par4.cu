#include <mpi.h>
#include <iostream>
#include <cmath>
#include <cstring>

#include <nvtx3/nvToolsExt.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

class parser{
public:
    parser(int argc, char** argv){
        this->_grid_size = 512;
        this->_accur = 1e-6;
        this->_iters = 1000000;
        for (int i=0; i<argc-1; i++){
            std::string arg = argv[i];
            if (arg == "-accur"){
                std::string dump = std::string(argv[i+1]);
                this->_accur = std::stod(dump);
            }else if (arg == "-a"){
                this->_grid_size = std::stoi(argv[i + 1]);
            }else if (arg == "-i"){
                this->_iters = std::stoi(argv[i + 1]);
            }
        }

    };
    __host__ double accuracy() const{
        return this->_accur;
    }
    __host__ int iterations() const{
        return this->_iters;
    }
    __host__ int grid()const{
        return this->_grid_size;
    }
private:
    double _accur;
    int _grid_size;
    int _iters;

};


double corners[4] = { 10, 20, 30, 20 };

__global__
void cross_calc(double* first_dev, double* second_dev, size_t size) {
    //получаем индексы

    size_t j = blockIdx.x;
    size_t i = threadIdx.x;
    //основные вычисления
    if (i != 0 && j != 0) {

        second_dev[j * size + i] = 0.25 * (
            first_dev[j * size + i - 1] +
            first_dev[j * size + i + 1] +
            first_dev[(j + 1) * size + i] +
            first_dev[(j - 1) * size + i]
            );

    }

}

__global__
void get_error_matrix(double* first_dev, double* second_dev, double* out) {
    //получаем индексы
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // максимальная ошибка
    if (blockIdx.x != 0 && threadIdx.x != 0) {

        out[idx] = std::abs(second_dev[idx] - first_dev[idx]);

    }

}


int main(int argc, char** argv) {
    parser input = parser(argc, argv);

    int size = input.grid();
    double min_error = input.accuracy();
    int max_iter = input.iterations();
    int full_size = size * size;
    double step = (corners[1] - corners[0]) / (size - 1);
    // инициализируем матрицы
    auto* first_dev = new double[size * size];
    auto* second_dev = new double[size * size];

    std::memset(first_dev, 0, sizeof(double) * size * size);

    first_dev[0] = corners[0];
    first_dev[size - 1] = corners[1];
    first_dev[size * size - 1] = corners[2];
    first_dev[size * (size - 1)] = corners[3];

    for (int i = 1; i < size - 1; i++) {
        first_dev[i] = corners[0] + i * step;
        first_dev[size * i] = corners[0] + i * step;
        first_dev[(size - 1) + size * i] = corners[1] + i * step;
        first_dev[size * (size - 1) + i] = corners[3] + i * step;
    }

    std::memcpy(second_dev, first_dev, sizeof(double) * full_size);

    // for (int i = 0; i < size; i ++) {
    //     for (int j = 0; j < size; j ++) {
    //         std::cout << first_dev[j * size + i] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;

    // выбираем устройство
    cudaSetDevice(3);

    double* dev_A, * dev_B, * dev_err, * dev_err_mat, * temp_stor = NULL;
    size_t tmp_stor_size = 0;
    // Распределение памяти
    cudaError_t status_A = cudaMalloc(&dev_A, sizeof(double) * full_size);
    cudaError_t status_B = cudaMalloc(&dev_B, sizeof(double) * full_size);
    cudaError_t status = cudaMalloc(&dev_err, sizeof(double));
    // возможные ошибки
    if (status != cudaSuccess) {
        std::cout << "Device error variable allocation error " << status << std::endl;
        return status;
    }
    status = cudaMalloc(&dev_err_mat, sizeof(double) * full_size);
    if (status != cudaSuccess) {
        std::cout << "Device error matrix allocation error " << status << std::endl;
        return status;
    }
    if (status_A != cudaSuccess) {
        std::cout << "Kernel A allocation error " << status << std::endl;
        return status;
    }
    else if (status_B != cudaSuccess) {
        std::cout << "Kernel B allocation error " << status << std::endl;
        return status;
    }

    status_A = cudaMemcpy(dev_A, first_dev, sizeof(double) * full_size, cudaMemcpyHostToDevice);
    if (status_A != cudaSuccess) {
        std::cout << "Kernel A copy to device error " << status << std::endl;
        return status_A;
    }
    status_B = cudaMemcpy(dev_B, second_dev, sizeof(double) * full_size, cudaMemcpyHostToDevice);
    if (status_B != cudaSuccess) {
        std::cout << "kernel B copy to device error " << status << std::endl;
        return status_B;
    }

    status = cub::DeviceReduce::Max(temp_stor, tmp_stor_size, dev_err_mat, dev_err, full_size);
    if (status != cudaSuccess) {
        std::cout << "Max reduction error " << status << std::endl;
        return status;
    }

    status = cudaMalloc(&temp_stor, tmp_stor_size);
    if (status != cudaSuccess) {
        std::cout << "Temporary storage allocation error " << status << std::endl;
        return status;
    }

    int i = 0;
    double error = 1.0;

    nvtxRangePushA("Main loop");
    // основной цикл
    while (i < max_iter && error > min_error) {
        i++;
      
        cross_calc<<<size - 1, size - 1 >>>(dev_A, dev_B, size);

        if (i % 100 == 0) {
            // получаем матрицу ошибок
            get_error_matrix<<<size - 1, size - 1 >>>(dev_A, dev_B, dev_err_mat);
            // находим максимальную ошибку
            cub::DeviceReduce::Max(temp_stor, tmp_stor_size, dev_err_mat, dev_err, full_size);
            // копируем память
            cudaMemcpy(&error, dev_err, sizeof(double), cudaMemcpyDeviceToHost);

        }
        // меняем матрицы
        std::swap(dev_A, dev_B);


    }

    nvtxRangePop();

    std::cout << "Error: " << error << std::endl;
    std::cout << "Iteration: " << i << std::endl;

    cudaFree(temp_stor);
    cudaFree(dev_err_mat);
    cudaFree(dev_A);
    cudaFree(dev_B);
    delete[] first_dev;
    delete[] second_dev;
    return 0;
}
