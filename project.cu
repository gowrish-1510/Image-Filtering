#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>  // Added for fminf

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Constants
#define TILE_SIZE 16

// CUDA Kernel: Grayscale
__global__ void convert_to_grayscale(const uchar4* input, uchar4* output, size_t rows, size_t cols) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= cols || y >= rows) return;
    int idx = y * cols + x;
    unsigned char luminance = (unsigned char)(0.3f * input[idx].x + 0.59f * input[idx].y + 0.11f * input[idx].z);
    output[idx] = make_uchar4(luminance, luminance, luminance, 255);
}

// CUDA Kernel: Sepia
__global__ void sepia(const uchar4* input, uchar4* output, size_t rows, size_t cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows) return;
    int idx = y * cols + x;
    uchar4 pixel = input[idx];
    float red   = (pixel.x * 0.393f) + (pixel.y * 0.769f) + (pixel.z * 0.189f);
    float green = (pixel.x * 0.349f) + (pixel.y * 0.686f) + (pixel.z * 0.168f);
    float blue  = (pixel.x * 0.272f) + (pixel.y * 0.534f) + (pixel.z * 0.131f);
    output[idx] = make_uchar4((unsigned char)fminf(red, 255.0f), 
                              (unsigned char)fminf(green, 255.0f), 
                              (unsigned char)fminf(blue, 255.0f), 
                              255);
}

// CUDA Kernel: Mirror (vertical flip only)
__global__ 
void mirror(const uchar4* const inputChannel, uchar4* outputChannel, int numRows, int numCols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= numCols || row >= numRows) {
        return;
    }
    
    // Vertical flip: mirror along vertical axis.
    int thread_x = col;
    int thread_y = row;
    int thread_x_new = numCols - thread_x - 1; // adjust for 0-indexing
    int thread_y_new = thread_y;
    
    int myId = row * numCols + col;
    int myId_new = thread_y_new * numCols + thread_x_new;
    outputChannel[myId_new] = inputChannel[myId];
}

// Host function for mirror filter (vertical flip only)
uchar4* mirror_ops(uchar4 *d_inputImageRGBA, size_t numRows, size_t numCols)
{
    // Set reasonable block size
    const dim3 blockSize(4, 4, 1);
    // Calculate grid size (make sure to cover all pixels)
    int gridCols = (numCols + blockSize.x - 1) / blockSize.x;
    int gridRows = (numRows + blockSize.y - 1) / blockSize.y;
    const dim3 gridSize(gridCols, gridRows, 1);

    const size_t numPixels = numRows * numCols;

    uchar4 *d_outputImageRGBA;
    cudaError_t err = cudaMalloc(&d_outputImageRGBA, sizeof(uchar4) * numPixels);
    if (err != cudaSuccess) {
        printf("CUDA memory allocation error in mirror_ops: %s\n", cudaGetErrorString(err));
        return NULL;
    }

    // Call mirror kernel (always vertical flip)
    mirror<<<gridSize, blockSize>>>(d_inputImageRGBA, d_outputImageRGBA, numRows, numCols);
    cudaDeviceSynchronize(); 

    // Allocate host memory for output image.
    uchar4* h_out = (uchar4*)malloc(sizeof(uchar4) * numPixels);
    if (h_out == NULL) {
        printf("Host memory allocation error in mirror_ops\n");
        cudaFree(d_inputImageRGBA);
        cudaFree(d_outputImageRGBA);
        return NULL;
    }

    // Copy output from device to host.
    err = cudaMemcpy(h_out, d_outputImageRGBA, sizeof(uchar4) * numPixels, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("CUDA memory copy error in mirror_ops: %s\n", cudaGetErrorString(err));
        free(h_out);
        cudaFree(d_inputImageRGBA);
        cudaFree(d_outputImageRGBA);
        return NULL;
    }

    // Cleanup device memory.
    cudaFree(d_inputImageRGBA); // Free input image as it's no longer needed.
    cudaFree(d_outputImageRGBA);

    return h_out;
}

// Image Processing Function for grayscale and sepia effects
uchar4* process_image(uchar4 *d_image, size_t rows, size_t cols, const char* effect_type) {
    uchar4 *d_result;
    cudaError_t err = cudaMalloc((void **) &d_result, rows * cols * sizeof(uchar4));
    if (err != cudaSuccess) {
        printf("CUDA memory allocation error: %s\n", cudaGetErrorString(err));
        return NULL;
    }

    dim3 block_size(TILE_SIZE, TILE_SIZE, 1);
    dim3 grid_size((cols + TILE_SIZE - 1) / TILE_SIZE, (rows + TILE_SIZE - 1) / TILE_SIZE, 1);
    
    if (strcmp(effect_type, "grayscale") == 0) {
        convert_to_grayscale<<<grid_size, block_size>>>(d_image, d_result, rows, cols);
    } 
    else if (strcmp(effect_type, "sepia") == 0) {
        sepia<<<grid_size, block_size>>>(d_image, d_result, rows, cols);
    }
    
    // Check for kernel execution errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
        cudaFree(d_result);
        return NULL;
    }
    
    uchar4* h_result = (uchar4*)malloc(rows * cols * sizeof(uchar4));
    if (h_result == NULL) {
        printf("Host memory allocation error\n");
        cudaFree(d_result);
        return NULL;
    }
    
    err = cudaMemcpy(h_result, d_result, rows * cols * sizeof(uchar4), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("CUDA memory copy error: %s\n", cudaGetErrorString(err));
        free(h_result);
        cudaFree(d_result);
        return NULL;
    }
    
    cudaFree(d_result);
    return h_result;
}

// Image loading function
uchar4* load_image_in_GPU(const char* filename, size_t* rows, size_t* cols) {
    int width, height, channels;
    
    // Load image directly as RGBA
    unsigned char* h_img = stbi_load(filename, &width, &height, &channels, 4);
    if (!h_img) {
        printf("Failed to load image: %s\n", filename);
        return NULL;
    }
    
    *cols = width;
    *rows = height;
    size_t numPixels = width * height;
    
    // Allocate device memory
    uchar4* d_img;
    cudaError_t err = cudaMalloc((void**)&d_img, sizeof(uchar4) * numPixels);
    if (err != cudaSuccess) {
        printf("CUDA memory allocation error: %s\n", cudaGetErrorString(err));
        stbi_image_free(h_img);
        return NULL;
    }
    
    // Copy image data to device
    err = cudaMemcpy(d_img, h_img, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CUDA memory copy error: %s\n", cudaGetErrorString(err));
        cudaFree(d_img);
        stbi_image_free(h_img);
        return NULL;
    }
    
    stbi_image_free(h_img);
    return d_img;
}

// Image saving function
void save_image_from_GPU(uchar4* h_image, const char* filename, size_t rows, size_t cols) {
    // h_image is in host memory
    int stride_in_bytes = cols * 4; // 4 bytes per pixel (RGBA)
    int success = stbi_write_png(filename, cols, rows, 4, h_image, stride_in_bytes);
    
    if (success) {
        printf("Successfully saved image: %s\n", filename);
    } else {
        printf("Failed to save image: %s\n", filename);
    }
}

// Main Function
int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage: %s <input_file> <effect_type>\n", argv[0]);
        printf("Available effects: grayscale, sepia, mirror\n");
        printf("For mirror effect, a vertical flip is performed by default.\n");
        return 1;
    }
    
    const char* input_file = argv[1];
    const char* effect_type = argv[2];
    
    if (strcmp(effect_type, "grayscale") != 0 && 
        strcmp(effect_type, "sepia") != 0 && 
        strcmp(effect_type, "mirror") != 0) {
        printf("Invalid effect type. Choose 'grayscale', 'sepia', or 'mirror'\n");
        return 1;
    }
    
    size_t rows, cols;
    uchar4 *d_in = load_image_in_GPU(input_file, &rows, &cols);
    if (d_in == NULL) {
        return 1;
    }
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Start timing
    cudaEventRecord(start, 0);
    
    uchar4 *h_out = NULL;
    if (strcmp(effect_type, "mirror") == 0) {
        // For mirror effect, we always perform a vertical flip.
        h_out = mirror_ops(d_in, rows, cols);
        // d_in has been freed inside mirror_ops.
        d_in = NULL;
    } else {
        h_out = process_image(d_in, rows, cols, effect_type);
    }
    
    // Stop timing
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Image processing time: %f ms\n", elapsedTime);
    
    // Clean up events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    if (h_out == NULL) {
        if(d_in) cudaFree(d_in);
        return 1;
    }
    
    // Generate output filename based on the effect type.
    char output_filename[100];
    sprintf(output_filename, "%s.png", effect_type);
    save_image_from_GPU(h_out, output_filename, rows, cols);
    
    free(h_out);
    if(d_in) cudaFree(d_in);
    return 0;
}
