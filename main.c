#include <stdio.h>
#include <math.h>
#include <stdarg.h>
#include <errno.h>
#include <fts.h>
#include <fnmatch.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#include "inc/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "inc/stb_image_write.h"

#define CHECK \
do { \
	printf("at %d\n", __LINE__); \
	fflush(stdout); 						 \
} while(0)

typedef struct {
	unsigned char *data;
	int w, h;
} Image;

Image newimage(int w, int h);
Image elbp(Image);
void tograyscale(Image);
unsigned char *pixelat(Image, int x, int y);
int entcmp(const FTSENT **, const FTSENT **);
void trainonfiles(int hist[], int n, char *dir, char *pattern);
void *emalloc(size_t num, size_t size);
void die(char *fmt, ...);

/*
 * training:
 * go through every picture, calculate a LQP, add to histogram
 * build list of binary patterns by filtering by threshold
 * do k-means clustering (probably multiple times, choose least err)
 * assign every possible binary number to its nearest neighbor,
 * create look up table
 *
 * to find similar faces, build histogram of code points
 * sort by cosine similarity on histogram
 *
 */

int 
main(int argc, char *argv[]) 
{
	int nfilestrained = 10;
	// 2^8 = 256
	// just for LBP -- increase for LQP to 2^24 or however many
	int *hist = emalloc(256, sizeof(int));
	trainonfiles(hist, nfilestrained, "./lfw-deepfunneled", "*.jpg");

	free(hist);
	hist = NULL;
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
		x = radius * cos(2.0*M_PI*n/neighbors);
		y = radius * -sin(2.0*M_PI*n/neighbors);
		fx = floor(x);
		fy = floor(y);
		cx = ceil(x);
		cy = ceil(y);
		ty = y - fy;
		tx = x - fx;
		// interp weights
		w1 = (1-tx)*(1-ty);
		w2 =    tx *(1-ty);
		w3 = (1-tx)*   ty;
		w4 =    tx *   ty;
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
	return out;
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
			r = p[0] * 0.30;
			g = p[1] * 0.59;
			b = p[2] * 0.11;
			v = r+g+b;
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

int 
entcmp(const FTSENT **a, const FTSENT **b)
{
	return strcmp((*a)->fts_name, (*b)->fts_name);
}

void 
trainonfiles(int hist[], int n, char *dir, char *pattern)
{
	FTS *tree;
	FTSENT *f;
	char *argv[] = { dir, NULL };
	int i = 0;
	char outname[128];

	tree = fts_open(argv, FTS_LOGICAL | FTS_NOSTAT, entcmp);
	if(!tree)
		die("fts_open failed");
	while(i < n && (f = fts_read(tree))) {
		switch(f->fts_info) {
		case FTS_DNR: /* cannot read directory */
		case FTS_ERR: /* misc error */
		case FTS_NS:  /* stat() error */
			fprintf(stderr, "error while reading %s\n", f->fts_path);
			continue;
		case FTS_DP:
			/* ignore post-order visit to directory */
			continue;
		}
		// check if matches pattern. FNM_PERIOD means *.c won't match .invis.c
		if(!fnmatch(pattern, f->fts_name, FNM_PERIOD)) {
			Image img;
			int comp;
			if(!(img.data = stbi_load(f->fts_path, &img.w, &img.h, &comp, 0)))
				die("could not load %s", f->fts_path);
			if(comp != 3)
				die("picture has %d components, should be 3", comp);

			tograyscale(img);

			// maybe make this in place to save memory
			Image out = elbp(img);

			strcpy(outname, "./outs/");
			strcat(outname, f->fts_name);
			if(!stbi_write_png(outname, out.w, out.h, comp, out.data, out.w*3))
				die("could not write %s", outname);

			free(img.data);
			img.data = NULL;
			free(out.data);
			out.data = NULL;
			i++;
		}
		if(f->fts_info == FTS_DC)
			fprintf(stderr, "%s: cycle in directory tree", f->fts_path);
	}
	printf("i is %d\n", i);
	if(errno)
		die("fts_read errno is set");
	if(fts_close(tree) < 0)
		die("fts_close failed");
}

void *
emalloc(size_t num, size_t size)
{
	void *mem = calloc(num, size);
	if(!mem)
		die("memory allocation failed");
	return mem;
}

void
die(char *fmt, ...)
{
	va_list vargs;
	va_start(vargs, fmt);
	vfprintf(stderr, fmt, vargs);
	fprintf(stderr, "\n");
	exit(1);
}
