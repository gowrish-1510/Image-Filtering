#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>  // For clock()

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Define a simple uchar4 type for CPU usage.
typedef struct {
    unsigned char x, y, z, w;
} uchar4;

// CPU Function: Grayscale and Sepia filters
uchar4* process_image_cpu(uchar4* input, size_t rows, size_t cols, const char* effect_type) {
    size_t numPixels = rows * cols;
    uchar4* output = (uchar4*)malloc(sizeof(uchar4) * numPixels);
    if (!output) {
        printf("Host memory allocation error\n");
        return NULL;
    }
    
    if (strcmp(effect_type, "grayscale") == 0) {
        // Convert each pixel to grayscale.
        for (size_t i = 0; i < numPixels; i++) {
            unsigned char luminance = (unsigned char)(0.3f * input[i].x +
                                                       0.59f * input[i].y +
                                                       0.11f * input[i].z);
            output[i].x = luminance;
            output[i].y = luminance;
            output[i].z = luminance;
            output[i].w = 255;
        }
    } 
    else if (strcmp(effect_type, "sepia") == 0) {
        // Apply sepia tone transformation.
        for (size_t i = 0; i < numPixels; i++) {
            uchar4 pixel = input[i];
            float red   = pixel.x * 0.393f + pixel.y * 0.769f + pixel.z * 0.189f;
            float green = pixel.x * 0.349f + pixel.y * 0.686f + pixel.z * 0.168f;
            float blue  = pixel.x * 0.272f + pixel.y * 0.534f + pixel.z * 0.131f;
            output[i].x = (unsigned char)(red   > 255.0f ? 255 : red);
            output[i].y = (unsigned char)(green > 255.0f ? 255 : green);
            output[i].z = (unsigned char)(blue  > 255.0f ? 255 : blue);
            output[i].w = 255;
        }
    }
    
    return output;
}

// CPU Function: Mirror filter (vertical flip only)
uchar4* mirror_ops_cpu(uchar4* input, size_t rows, size_t cols) {
    size_t numPixels = rows * cols;
    uchar4* output = (uchar4*)malloc(sizeof(uchar4) * numPixels);
    if (!output) {
        printf("Host memory allocation error in mirror_ops_cpu\n");
        return NULL;
    }
    
    for (size_t row = 0; row < rows; row++) {
        for (size_t col = 0; col < cols; col++) {
            size_t src_idx = row * cols + col;
            size_t dst_idx = row * cols + (cols - col - 1); // Vertical flip: mirror along vertical axis.
            output[dst_idx] = input[src_idx];
        }
    }
    
    return output;
}

// CPU Function: Load image and convert it into an array of uchar4.
uchar4* load_image_cpu(const char* filename, size_t* rows, size_t* cols) {
    int width, height, channels;
    unsigned char* h_img = stbi_load(filename, &width, &height, &channels, 4);
    if (!h_img) {
        printf("Failed to load image: %s\n", filename);
        return NULL;
    }
    
    *cols = width;
    *rows = height;
    size_t numPixels = width * height;
    
    // Allocate memory for our uchar4 array.
    uchar4* image = (uchar4*)malloc(sizeof(uchar4) * numPixels);
    if (!image) {
        printf("Host memory allocation error in load_image_cpu\n");
        stbi_image_free(h_img);
        return NULL;
    }
    memcpy(image, h_img, sizeof(uchar4) * numPixels);
    stbi_image_free(h_img);
    
    return image;
}

// CPU Function: Save image from an array of uchar4.
void save_image_cpu(uchar4* image, const char* filename, size_t rows, size_t cols) {
    int stride_in_bytes = cols * 4; // 4 bytes per pixel (RGBA)
    int success = stbi_write_png(filename, cols, rows, 4, image, stride_in_bytes);
    
    if (success) {
        printf("Successfully saved image: %s\n", filename);
    } else {
        printf("Failed to save image: %s\n", filename);
    }
}

// Main function
int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage: %s <input_file> <effect_type>\n", argv[0]);
        printf("Available effects: grayscale, sepia, mirror\n");
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
    uchar4* input_image = load_image_cpu(input_file, &rows, &cols);
    if (!input_image) {
        return 1;
    }
    
    // Start timing the processing
    clock_t start_time = clock();
    
    uchar4* output_image = NULL;
    if (strcmp(effect_type, "mirror") == 0) {
        output_image = mirror_ops_cpu(input_image, rows, cols);
    } else {
        output_image = process_image_cpu(input_image, rows, cols, effect_type);
    }
    
    // Stop timing the processing
    clock_t end_time = clock();
    double processing_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Image processing time: %f seconds\n", processing_time);
    
    if (!output_image) {
        free(input_image);
        return 1;
    }
    
    // Generate output filename based on the effect type.
    char output_filename[100];
    sprintf(output_filename, "%s.png", effect_type);
    save_image_cpu(output_image, output_filename, rows, cols);
    
    free(input_image);
    free(output_image);
    
    return 0;
}
