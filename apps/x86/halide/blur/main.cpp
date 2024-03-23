#include "blur/blur.h"
#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <png.h>
#include <vector>

bool read_png_file(const char *filename, std::vector<uint16_t> &buffer,
    int &width, int &height) {
  FILE *fp = fopen(filename, "rb");
  if (!fp)
    return false;

  png_structp png =
      png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (!png)
    return false;

  png_infop info = png_create_info_struct(png);
  if (!info)
    return false;

  if (setjmp(png_jmpbuf(png)))
    return false;

  png_init_io(png, fp);
  png_read_info(png, info);

  width = png_get_image_width(png, info);
  height = png_get_image_height(png, info);
  buffer.resize(width * height);

  png_bytep row = new png_byte[width];

  for (int y = 0; y < height; y++) {
    png_read_row(png, row, NULL);
    for (int x = 0; x < width; x++) {
      buffer[y * width + x] = row[x];
    }
  }

  fclose(fp);

  if (png && info)
    png_destroy_read_struct(&png, &info, NULL);

  delete[] row;

  return true;
}

bool write_png_file(
    const char *filename, const uint16_t *buffer, int width, int height) {
  FILE *fp = fopen(filename, "wb");
  if (!fp)
    return false;

  png_structp png =
      png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (!png)
    return false;

  png_infop info = png_create_info_struct(png);
  if (!info)
    return false;

  if (setjmp(png_jmpbuf(png)))
    return false;

  png_init_io(png, fp);

  // Bit depth is usually 8 for grayscale images, 1/2/4/16 are also possible.
  png_set_IHDR(png, info, width, height, 8, PNG_COLOR_TYPE_GRAY,
      PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

  png_write_info(png, info);

  png_bytep row = new png_byte[width];

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      row[x] = buffer[y * width + x];
    }
    png_write_row(png, row);
  }

  png_write_end(png, NULL);
  fclose(fp);

  if (png && info)
    png_destroy_write_struct(&png, &info);

  delete[] row;

  return true;
}

typedef void (*blurtype)(void *ctxt, int_fast32_t n, int_fast32_t m,
    uint16_t *g, const uint16_t *inp);

int exec_blur(blurtype func, std::string output_name, int width, int height,
    uint16_t *img_in, uint16_t *img_out) {
  auto start = std::chrono::steady_clock::now();
  int iterations = 100;
  for (int i = 0; i < iterations; i++) {
    func(nullptr, width, height, img_out, img_in);
  }
  auto stop = std::chrono::steady_clock::now();
  float time = (float)std::chrono::duration_cast<std::chrono::microseconds>(
      (stop - start) / iterations)
                   .count();
  printf("%s: %f microseconds\n", output_name.c_str(), time);

  std::string file_name = output_name + std::string(".png");
  if (!write_png_file(file_name.c_str(), img_out, width, height)) {
    std::cerr << "Error writing PNG file." << std::endl;
  }

  return 0;
}

int main() {
  const char *read_file = "gray_scaled.png";
  std::vector<uint16_t> buffer;
  int width, height;

  if (read_png_file(read_file, buffer, width, height)) {
    printf("width: %d\n", (int)width);
    printf("height: %d\n", (int)height);

    std::vector<uint16_t> img_in((width + 2) * (height + 2));
    for (int i = 0; i < height; i++) {
      memcpy(&img_in[(i + 1) * (width + 2) + 1], &buffer[i * width],
          sizeof(uint16_t) * width);
    }

    std::vector<uint16_t> ref_img_out(width * height);
    std::vector<uint16_t> img_out(width * height);
    exec_blur(blur, "blur", width, height, img_in.data(), ref_img_out.data());
    exec_blur(exo_blur_halide, "exo_blur_halide", width, height, img_in.data(),
        img_out.data());

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        if (ref_img_out[y * width + x] != img_out[y * width + x]) {
          std::cerr << "Mismatch at (" << x << ", " << y
                    << "): " << ref_img_out[y * width + x]
                    << " != " << img_out[y * width + x] << std::endl;
          return 1;
        }
      }
    }

    std::cout << "Results match!" << std::endl;
  } else {
    std::cerr << "Error reading PNG file." << std::endl;
  }

  return 0;
}
