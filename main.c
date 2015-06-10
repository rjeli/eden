#include <stdio.h>
#include <math.h>
#include <stdarg.h>

#define STB_IMAGE_IMPLEMENTATION
#include "inc/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "inc/stb_image_write.h"

struct Image {
	unsigned char *data;
	int w, h;
};
typedef struct Image Image;

Image elbp(Image);
void tograyscale(Image);
unsigned char *pixelat(Image, int x, int y);
void *emalloc(size_t num, size_t size);
void die(char *fmt, ...);

int 
main(int argc, char *argv[]) 
{
  #pragma omp parallel for
	for(int n=0; n<2; ++n)
		printf("%d\n", n);

	Image img;
	int n;
	img.data = stbi_load("stannis.jpg", &img.w, &img.h, &n, 0);

	if(n != 3) {
		fprintf(stderr, "picture has %d components, should be 3", n);
		return -1;
	}

	tograyscale(img);

	Image out = elbp(img);

	stbi_write_png("stannis.png", out.w, out.h, n, out.data, out.w*3);

	free(img.data);
	img.data = NULL;
	free(out.data);
	out.data = NULL;

	return EXIT_SUCCESS;
}

void
tograyscale(Image img)
{
	unsigned int i, j;
	unsigned char *p, v;
	float r, g, b;

	for(i=0; i<img.h; ++i) {
		for(j=0; j<img.w; ++j) {
			p = pixelat(img, j, i);
			r = (float)p[0] * 0.30;
			g = (float)p[1] * 0.59;
			b = (float)p[2] * 0.11;
			v = (unsigned char)(r+g+b);
			p[0] = v;
			p[1] = v;
			p[2] = v;
		}
	}
}

Image
elbp(Image img)
{
	unsigned int neighbors = 8;
	unsigned int radius = 1;
	unsigned int n, i, j;
	float x, y, fx, fy, cx, cy, tx, ty;
	float w1, w2, w3, w4;
	float t;
	unsigned char add, *p;

	Image out;
	out.w = img.w;
	out.h = img.h;
	out.data = emalloc(out.w*out.h, sizeof(unsigned char)*3);

	for(n=0; n<neighbors; ++n) {
		x =  (float)radius * cos(2.0 * M_PI * (float)n / (float)neighbors);
		y = -(float)radius * sin(2.0 * M_PI * (float)n / (float)neighbors);

		fx = (int)floor(x);
		fy = (int)floor(y);
		cx = (int)ceil(x);
		cy = (int)ceil(y);

		ty = y - (float)fy;
		tx = x - (float)fx;

		// interp weights
		w1 = (1.0-tx) * (1.0-ty);
		w2 =      tx  * (1.0-ty);
		w3 = (1.0-tx) *      ty;
		w4 =      tx  *      ty;

		for(i=radius; i<img.h-radius; ++i) {
			for(j=radius; j<img.w-radius; ++j) {
				t = w1 * (float)*pixelat(img, j+fx, i+fy) +
						w2 * (float)*pixelat(img, j+cx, i+fy) +
						w3 * (float)*pixelat(img, j+fx, i+cy) +
						w4 * (float)*pixelat(img, j+cx, i+cy);
				add = (t > *pixelat(img, j, i)) << n;
				p = pixelat(out, j-radius, i-radius);
				// set r, g, b
				p[0] += add;
				p[1] += add;
				p[2] += add;
			}
		}
	}

	return out;
}

unsigned char *
pixelat(Image img, int x, int y)
{
	return img.data + 3*(y*img.w+x);
}

void *
emalloc(size_t num, size_t size)
{
	void *mem = calloc(num, size);
	if(mem == NULL)
		die("memory allocation failed");
	return mem;
}

void
die(char *fmt, ...)
{
	va_list vargs;
	va_start(vargs, fmt);
	vfprintf(stderr, fmt, vargs);
	fprintf(stderr, ".\n");
	exit(1);
}
