#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <cub/cub.cuh>

#define BILLION 1000000000.0
#define MAX(x, y) (((x)>(y))?(x):(y)) //максимум 
#define ABS(x) ((x)<0 ? -(x): (x)) //модуль
#define GETID(i,j,ld) ((((j)-1)*(ld))+((i)-1)) //вычисление индекса по строке и столбцу

double CORNER_1 = 10;
double CORNER_2 = 20;
double CORNER_3 = 30;
double CORNER_4 = 20;

__global__ void fillBorders(double* arr, double top, double bottom, double left, double right, int m) {

    // Выполняем линейную интерполяцию на границах массива
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    if ((j > 0) && (j < m)) {
        arr[GETID(1, j, m)] = arr[GETID(1, j + m, m)] = (arr[GETID(1, 1, m)] + top * (j - 1));   //top
        arr[GETID(m, j, m)] = arr[GETID(m, j + m, m)] = (arr[GETID(m, 1, m)] + bottom * (j - 1)); //bottom
        arr[GETID(j, 1, m)] = arr[GETID(j, m + 1, m)] = (arr[GETID(1, 1, m)] + left * (j - 1)); //left
        arr[GETID(j, m, m)] = arr[GETID(j, 2 * m, m)] = (arr[GETID(1, m, m)] + right * (j - 1)); //right
    }
}
__global__ void getAverage(double* arr, int p, int q, int m) {

    // Присваиваем ячейке среднее значение от креста, окружающего её

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if ((i > 1) && (i < m) && (j > 1) && (j < m)) {
        arr[GETID(i, j + p, m)] = 0.25 * (arr[GETID(i + 1, j + q, m)]
            + arr[GETID(i - 1, j + q, m)]
            + arr[GETID(i, j - 1 + q, m)]
            + arr[GETID(i, j + 1 + q, m)]);
    }
}
__global__ void subtractArrays(const double* arr_a, double* arr_b, int m) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if ((i > 1) && (i < m) && (j > 1) && (j < m)) {
        arr_b[GETID(i, j, m)] = ABS(arr_a[GETID(i, j, m)] - arr_a[GETID(i, j + m, m)]);
    }
}


int main(int argc, char* argv[]) {
    struct timespec start, stop;
    clock_gettime(CLOCK_REALTIME, &start);
    double delta, min_error;
    int m, iter_max;

    // Проверка ввода данных
    if (argc < 4) {
        printf("%s\n", "Not enough args\n");
        exit(1);
    }
    else {
        m = atoi(argv[1]); // Размер сетки
        if (m == 0) {
            printf("%s\n", "Incorrect first parametr\n");
            exit(1);
        }
        iter_max = atoi(argv[2]); // Количество итераций
        if (iter_max == 0) {
            printf("%s\n", "Incorrect second parametr\n");
            exit(1);
        }
        min_error = atof(argv[3]); // Точность
        if (min_error == 0) {
            printf("%s\n", "Incorrect third parametr\n");
            exit(1);
        }
    }

    int iter = 0;
    double err = min_error + 1;
    size_t size = 2 * m * m * sizeof(double);
    double* arr = (double*)malloc(size);

    for (int j = 1; j <= m; j++) {
        for (int i = 1; i <= m; i++) {
            arr[GETID(i, j, m)] = 0;
        }
    }

    arr[GETID(1, 1, m)] = arr[GETID(1, m + 1, m)] = CORNER_1;
    arr[GETID(1, m, m)] = arr[GETID(1, 2 * m, m)] = CORNER_2;
    arr[GETID(m, 1, m)] = arr[GETID(m, m + 1, m)] = CORNER_4;
    arr[GETID(m, m, m)] = arr[GETID(m, 2 * m, m)] = CORNER_3;

    // Коэффициенты для линейной интерполяции
    double top, bottom, left, right;

    top = (arr[GETID(1, m, m)] - arr[GETID(1, 1, m)]) / (m - 1);
    bottom = (arr[GETID(m, m, m)] - arr[GETID(m, 1, m)]) / (m - 1);
    left = (arr[GETID(m, 1, m)] - arr[GETID(1, 1, m)]) / (m - 1);
    right = (arr[GETID(m, m, m)] - arr[GETID(1, m, m)]) / (m - 1);

    cudaError_t cudaErr = cudaSuccess;
    double* d_A = NULL;
    cudaErr = cudaMalloc((void**)&d_A, size);
    if (cudaErr != cudaSuccess) {
        fprintf(stderr, "(error code %s)!\n", cudaGetErrorString(cudaErr));
        exit(1);
    }

    double* d_B = NULL;
    cudaErr = cudaMalloc((void**)&d_B, size / 2);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaErr = cudaMemcpyAsync(d_A, arr, size, cudaMemcpyHostToDevice, stream);

    if (cudaErr != cudaSuccess) {
        fprintf(stderr, "(error code %s)!\n", cudaGetErrorString(cudaErr));
        exit(1);
    }

    // заполняем границы с помощью линейной интерполяции
    fillBorders <<<(m + 1024 - 1) / 1024, 1024, 0, stream>>> (d_A, top, bottom, left, right, m);
    cudaErr = cudaMemcpyAsync(arr, d_A, size, cudaMemcpyDeviceToHost, stream);
    if (cudaErr != cudaSuccess) {
        fprintf(stderr, "(error code %s)!\n", cudaGetErrorString(cudaErr));
        exit(1);
    }

    printf("\n");
    if (m == 13) {
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= m; j++) {
                printf("%06.3lf ", arr[GETID(i, j, m)]);
            }
            printf("\n");
        }
    }

    int p = m, q = 0, flag = 1;
    double* h_buff = (double*)malloc(sizeof(double));
    double* d_buff = NULL;

    cudaErr = cudaMalloc((void**)&d_buff, sizeof(double));
    if (cudaErr != cudaSuccess) {
        fprintf(stderr, "(error code %s)!\n", cudaGetErrorString(cudaErr));
        exit(1);
    }

    dim3 grid((m + 32 - 1) / 32, (m + 32 - 1) / 32);
    dim3 block(32, 32);

    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    // Вызываем DeviceReduce здесь, чтобы проверить, сколько памяти нам нужно для временного хранилища
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_B, d_buff, m * m, stream);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    bool graphCreated = false;
    cudaGraph_t graph;
    cudaGraphExec_t instance;

    nvtxRangePushA("Main loop");
    {
        while (iter < iter_max && flag) {
            if (!graphCreated) {
                // фиксируем вызовы ядра в графе перед их вызовом.
                cudaErr = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

                if (cudaErr != cudaSuccess) {
                    fprintf(stderr,
                        "Failed to start stream capture (error code %s)!\n",
                        cudaGetErrorString(cudaErr));
                    exit(1);
                }
                for (int i = 0; i < 100; i++) {
                    //q и p выбирают, какой массив мы считаем новым, а какой — старым.
                    q = (i % 2) * m;
                    p = m - q;
                    getAverage <<<grid, block, 0, stream>>> (d_A, p, q, m);
                }
                cudaErr = cudaStreamEndCapture(stream, &graph);
                if (cudaErr != cudaSuccess) {
                    fprintf(stderr,
                        "Failed to end stream capture (error code %s)!\n",
                        cudaGetErrorString(cudaErr));
                    exit(1);
                }
                cudaErr = cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
                if (cudaErr != cudaSuccess) {
                    fprintf(stderr,
                        "Failed to instantiate cuda graph (error code %s)!\n",
                        cudaGetErrorString(cudaErr));
                    exit(1);
                }
                graphCreated = true;
            }

            cudaErr = cudaGraphLaunch(instance, stream);
            if (cudaErr != cudaSuccess) {
                fprintf(stderr,
                    "Failed to launch cuda graph (error code %s)!\n",
                    cudaGetErrorString(cudaErr));
                exit(1);
            }
            if (cudaErr != cudaSuccess) {
                fprintf(stderr,
                    "Failed to synchronize the stream (error code %s)!\n",
                    cudaGetErrorString(cudaErr));
                exit(1);
            }

            // Проверяем ошибку каждые 100 итераций
            iter += 100;

            // вычисляем абсолютные значения различий массивов
            // а затем находим максимальную разницу(ошибку)
            subtractArrays <<<grid, block, 0, stream>>> (d_A, d_B, m);
            cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_B, d_buff, m * m, stream);
            cudaErr = cudaMemcpyAsync(h_buff, d_buff, sizeof(double), cudaMemcpyDeviceToHost, stream);

            if (cudaErr != cudaSuccess) {
                fprintf(stderr,
                    "Failed to copy error back to host memory(error code %s)!\n",
                    cudaGetErrorString(cudaErr));
                exit(1);
            }
            err = *h_buff;
            flag = err > min_error;
        }
    }

    nvtxRangePop();

    clock_gettime(CLOCK_REALTIME, &stop);
    delta = (stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec) / (double)BILLION;

    printf("Elapsed time %lf\n", delta);
    printf("Final result: %d, %0.8lf\n", iter, err);

    cudaErr = cudaMemcpy(arr, d_A, size, cudaMemcpyDeviceToHost);

    if (m == 13) {
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= m; j++) {
                printf("%06.3lf ", arr[GETID(i, j, m)]);
            }
            printf("\n");
        }
    }
    //освобождаем память
    free(arr);
    free(h_buff);
    cudaDestroy(stream);
    cudaFree(d_buff);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_temp_storage);

    return 0;
}
