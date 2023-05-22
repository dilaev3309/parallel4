#include <iostream>
#include <cmath>
#include <cstring>
#include <time.h>
#include <iomanip>
#include <nvtx3/nvToolsExt.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

double first_corner = 10;
double second_corner = 20;
double third_corner = 30;
double fourth_corner = 20;

//Вычисление новых значений элементов массивов

__global__
void cross_calc(double* A, double* Anew, size_t size){
    // get the block and thread indices
    
    size_t j = blockIdx.x;
    size_t i = threadIdx.x;
    // main cross computation. the average of 4 incident cells is taken
    if (i != 0 && j != 0){
       
        Anew[j * size + i] = 0.25 * (
            A[j * size + i - 1] + 
            A[j * size + i + 1] + 
            A[(j + 1) * size + i] + 
            A[(j - 1) * size + i]
        );
    
    }

}

//Вычисление ошибки между элементами массивов

__global__
void get_error_matrix(double* A, double* Anew, double* out){
    // Получаю индекс
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Получаю макс. ошибку
    if (blockIdx.x != 0 && threadIdx.x != 0){
        out[idx] = std::abs(Anew[idx] - A[idx]);
    }

}

int main(int argc, char ** argv){
    int max_iter, size;
    double min_error;

    // Проверка ввода данных
    if (argc < 4){
        std::cout << "Not enough args\n" << std::endl;
        exit(1);
    } else{
        size = atoi(argv[1]); // Размер сетки
        if (size == 0){
            std::cout <<"Incorrect parametr\n" << std::endl;
            exit(1);
        }
        max_iter = atoi(argv[2]); // Количество итераций
        if (max_iter == 0){
            std::cout <<"Incorrect parametr\n" << std::endl;
            exit(1);
        }
        min_error = atof(argv[3]); // Точность
        if (min_error == 0){
            std::cout <<"Incorrect parametr\n"<< std::endl;
            exit(1);
        }
    }

    clock_t a = clock();

    int full_size = size * size;
    double step = (second_corner - first_corner) / (size - 1);

    // Инициализация массивов
    auto* A = new double[size * size];
    auto* Anew = new double[size * size];

    std::memset(A, 0, sizeof(double) * size * size);

    // Угловые значения
    A[0] = first_corner;
    A[size - 1] = second_corner;
    A[size * size - 1] = third_corner;
    A[size * (size - 1)] = fourth_corner;

    // Значения краёв сетки
    for (int i = 1; i < size - 1; i ++) {
        A[i] = first_corner + i * step;
        A[size * i] = first_corner + i * step;
        A[(size-1) + size * i] = second_corner + i * step;
        A[size * (size-1) + i] = fourth_corner + i * step;
    }

    std::memcpy(Anew, A, sizeof(double) * full_size);

    //for (int i = 0; i < size; i ++) {
    //    for (int j = 0; j < size; j ++) {
    //        std::cout << std::fixed << std::setprecision(5) << A[j * size + i] << " ";
    //    }
    //    std::cout << std::endl;
    //}
    
    // Выбор девайса
    cudaSetDevice(3);
    
    double* dev_A, *dev_B, *dev_err, *dev_err_mat, *temp_stor = NULL;
    size_t tmp_stor_size = 0;

    // Выделение памяти на 2 матрицы и переменная ошибки 
    cudaError_t status_A = cudaMalloc(&dev_A, sizeof(double) * full_size);
    cudaError_t status_B = cudaMalloc(&dev_B, sizeof(double) * full_size);
    cudaError_t status = cudaMalloc(&dev_err, sizeof(double));

    // выявление ошибок с памятью
    if (status != cudaSuccess){
        std::cout << "Device error variable allocation error " << status << std::endl;
        return status;
    }

    // Выделение памяти на устройстве для матрицы ошибок
    status = cudaMalloc(&dev_err_mat, sizeof(double) * full_size);
    if (status != cudaSuccess){
        std::cout << "Device error matrix allocation error " << status << std::endl;
        return status;
    }
    if (status_A != cudaSuccess){
        std::cout << "Kernel A allocation error " << status << std::endl;
        return status;
    } else if (status_B != cudaSuccess){
        std::cout << "Kernel B allocation error " << status << std::endl;
        return status;
    }

    status_A = cudaMemcpy(dev_A, A, sizeof(double) * full_size, cudaMemcpyHostToDevice);
    if (status_A != cudaSuccess){
        std::cout << "Kernel A copy to device error " << status << std::endl;
        return status_A;
    }
    status_B = cudaMemcpy(dev_B, Anew, sizeof(double) * full_size, cudaMemcpyHostToDevice);
    if (status_B != cudaSuccess){
        std::cout << "kernel B copy to device error " << status << std::endl;
        return status_B;
    }

    status = cub::DeviceReduce::Max(temp_stor, tmp_stor_size, dev_err_mat, dev_err, full_size);
    if (status != cudaSuccess){
        std::cout << "Max reduction error " << status << std::endl;
        return status;
    }

    status = cudaMalloc(&temp_stor, tmp_stor_size);
    if (status != cudaSuccess){
        std::cout << "Temporary storage allocation error " << status  << std::endl;
        return status;
    }

    int i = 0;
    double error = 1.0;

    nvtxRangePushA("Main loop");

    // Основной алгоритм
    while (i < max_iter && error > min_error){
        i++;
        // Вычисление итерации
        cross_calc<<<size-1, size-1>>>(dev_A, dev_B, size);

        if (i % 100 == 0){
            // Получение ошибки
            // кол-во потоков = (size-1)^2
            get_error_matrix<<<size - 1, size - 1>>>(dev_A, dev_B, dev_err_mat);
            
            // Находим максимальную ошибку
            // Результат в dev_err
            cub::DeviceReduce::Max(temp_stor, tmp_stor_size, dev_err_mat, dev_err, full_size);
            
            // копирование ошибки в память хоста
            cudaMemcpy(&error, dev_err, sizeof(double), cudaMemcpyDeviceToHost);

        }

        // Смена массивов
        std::swap(dev_A, dev_B);

    }

    nvtxRangePop();

    // Вывод массивов
    cudaMemcpy(A, dev_A, sizeof(double) * full_size, cudaMemcpyDeviceToHost);

    std::cout << std::endl;

    //for (int i = 0; i < size; i ++) {
    //    for (int j = 0; j < size; j ++) {
    //        std::cout << A[j * size + i] << " ";
    //    }
    //    std::cout << std::endl;
    //}

    // Вывод результатов
    clock_t b = clock();
    double d = (double)(b-a)/CLOCKS_PER_SEC; // перевожу в секунды 
    std::cout << "Error: " << error << std::endl;
    std::cout << "Iteration: " << i << std::endl;
    std::cout << "Time: " << d << std::endl;

    // Очистка
    cudaFree(temp_stor);
    cudaFree(dev_err_mat);
    cudaFree(dev_A);
    cudaFree(dev_B);
    
    delete[] A;
    delete[] Anew;
    return 0;
}
