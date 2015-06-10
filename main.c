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

Image newimage(int w, int h);
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

Image
elbp(Image img)
{
	unsigned int neighbors = 8;
	unsigned int radius = 1;
	unsigned int n, i, j;
	double x, y, tx, ty;
	int fx, fy, cx, cy;
	double w1, w2, w3, w4;
	double t;
	unsigned char add, *p;

	Image out = newimage(img.w, img.h);

	for(n=0; n<neighbors; ++n) {
		x = radius * cos(2.0 * M_PI * n / neighbors);
		y = radius * -sin(2.0 * M_PI * n / neighbors);
		fx = floor(x);
		fy = floor(y);
		cx = ceil(x);
		cy = ceil(y);
		ty = y - fy;
		tx = x - fx;
		// interp weights
		w1 = (1.0-tx) * (1.0-ty);
		w2 =      tx  * (1.0-ty);
		w3 = (1.0-tx) *      ty;
		w4 =      tx  *      ty;
		for(i=radius; i<img.h-radius; ++i) {
			for(j=radius; j<img.w-radius; ++j) {
				t = w1 * *pixelat(img, j+fx, i+fy) +
						w2 * *pixelat(img, j+cx, i+fy) +
						w3 * *pixelat(img, j+fx, i+cy) +
						w4 * *pixelat(img, j+cx, i+cy);
				add = (t > *pixelat(img, j, i)) << n;
				p = pixelat(out, j-radius, i-radius);
				// r, g, b
				p[0] += add;
				p[1] += add;
				p[2] += add;
			}
		}
	}
	return out;
}

Image
newimage(int w, int h)
{
	Image out;
	out.w = w;
	out.h = h;
	out.data = emalloc(w*h, sizeof(unsigned char)*3);
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
