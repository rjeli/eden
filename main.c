#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdarg.h>
#include <errno.h>
#include <fts.h>
#include <fnmatch.h>
#include <float.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "inc/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "inc/stb_image_write.h"

#define CHECK \
do { \
	printf("at %d\n", __LINE__); \
	fflush(stdout); 						 \
} while(0)

int K = 150;

typedef struct {
	unsigned char *data;
	int w, h;
} Image;

uint32_t *lqp(Image);
void printbinary(int, void *);
Image lqptoimage(uint32_t *, int w, int h);
void tograyscale(Image);
unsigned char *pixelat(Image, int x, int y);
int entcmp(const FTSENT **, const FTSENT **);
void trainonfiles(int hist[], int n, char *dir, char *pattern);
int *kmeans(int *data, long n, int k, double **centroids, double *err);
double l2dist(uint32_t, double *);
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
	long i, j;
 	long sum = 0, tensum = 0;
	int nfilestrained = 100;
	int **labels;
	double ***centroidsets;
	double *errs;
	double minerr;
	int minindex;
	int *hist;

	srand(time(NULL));
 
	/* 2^24 = 16777216 */
	hist	= emalloc(16777216, sizeof(int));

	centroidsets = emalloc(10, sizeof(double **));
	errs = emalloc(10, sizeof(double));
	labels = emalloc(10, sizeof(int *));
	for(i=0; i<10; ++i) {
		centroidsets[i] = emalloc(K, sizeof(double *));
		for(j=0; j<K; ++j)
			centroidsets[i][j] = emalloc(24, sizeof(double));
	}

	trainonfiles(hist, nfilestrained, "./lfw-deepfunneled", "*.jpg");

	for(i=0; i<16777216; ++i) {
		if(hist[i]) 
			sum++;
		if(hist[i] > 10)
			tensum++;
	}

	printf("%ld patterns\n", sum);
	printf("%ld occur more than 10 times\n", sum);

	#pragma omp parallel for
	for(i=0; i<10; ++i) {
		labels[i] = kmeans(hist, 16777216, K, centroidsets[i], &errs[i]);
		printf("finished k-means %ld\n", i);
		fflush(stdout);
	}

	minerr = DBL_MAX;
	for(i=0; i<10; ++i) {
		if(errs[i]<minerr) {
			minerr = errs[i];
			minindex = i;
		}
	}

	printf("best k-means is %d, with error %f\n", minindex, errs[minindex]);
	

	Image img;
	int comp;
	if(!(img.data = stbi_load("stannis.jpg", &img.w, &img.h, &comp, 0)))
		die("could not load stannis.jpg");
	if(comp != 3)
		die("picture has %d components, should be 3", comp);

	tograyscale(img);

	uint32_t *outlqp = lqp(img);

	int x, y;
	uint32_t p;
	unsigned char *px;
	for(y=0; y<img.h; ++y) {
		for(x=0; x<img.w; ++x) {
			p = outlqp[y*img.w+x];
			for(int i=0; i<K; ++i) {
				if(l2dist(p, centroidsets[minindex][i]) < 0.8) {
					px = pixelat(img, x, y);
					px[0] = 255;
					px[1] = 0;
					px[2] = 0;
				}
			}
		}
	}

	if(!stbi_write_png("patterns.png", img.w, img.h, comp, img.data, img.w*3))
		die("could not write patterns.png");

	free(img.data);
	free(outlqp);

	free(hist);
	for(i=0; i<10; ++i) {
		for(j=0; j<K; ++j) {
			free(centroidsets[i][j]);
		}
		free(centroidsets[i]);
		free(labels[i]);
	}
	free(centroidsets);
	free(errs);
	free(labels);
	return EXIT_SUCCESS;
}

void 
printbinary(int s, void* p)
{
	int i, j;
	for(i=s-1; i>=0; i--)
		for(j=7; j>=0; j--)
			printf("%u",(*((unsigned char*)p+i)&(1<<j))>>j);
	puts("");
}

uint32_t *
lqp(Image img)
{
	/* inner ring - radius 1, 8 neighbors */
	/* outer ring - radius 2, 16 neighbors */
	unsigned int radius, n;
	unsigned int i, j;
	double x, y, tx, ty;
	int fx, fy, cx, cy;
	double w1, w2, w3, w4;
	double t;
	uint32_t add;

	uint32_t *out = emalloc(img.w*img.h, sizeof(uint32_t));

	for(radius=1; radius<3; ++radius) {
		for(n=0; n<radius*8; ++n) {
			x = radius * cos(2.0*M_PI*n/(radius*8));
			y = radius * -sin(2.0*M_PI*n/(radius*8));
			fx = floor(x);
			fy = floor(y);
			cx = ceil(x);
			cy = ceil(y);
			ty = y - fy;
			tx = x - fx;
			/* interp weights */
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
					add = (t > *pixelat(img, j, i)) << ((radius-1)*8 + n);
					out[(i-radius)*img.w+(j-radius)] += add;
				}
			}
		}
	}
	return out;
}

Image 
lqptoimage(uint32_t *data, int w, int h)
{
	int i, j;
	unsigned char *p, v;
	Image out;
	out.w = w;
	out.h = h;
	out.data = emalloc(w*h, sizeof(unsigned char)*3);

	for(i=0; i<h; ++i)
		for(j=0; j<w; ++j){
			p = pixelat(out, j, i);
			v = (unsigned char)(data[i*w+j]);
			p[0] = v;
			p[1] = v;
			p[2] = v;
		}
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
	int processed = 0;

	int i, j;
	int comp;
	Image img;
	/*char outname[128];*/
	uint32_t *outlqp;

	tree = fts_open(argv, FTS_LOGICAL | FTS_NOSTAT, entcmp);
	if(!tree)
		die("fts_open failed");
	while(processed < n && (f = fts_read(tree))) {
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
		/* check if matches pattern. FNM_PERIOD means *.c won't match .invis.c */
		if(!fnmatch(pattern, f->fts_name, FNM_PERIOD)) {
			if(!(img.data = stbi_load(f->fts_path, &img.w, &img.h, &comp, 0)))
				die("could not load %s", f->fts_path);
			if(comp != 3)
				die("picture has %d components, should be 3", comp);

			tograyscale(img);

			outlqp = lqp(img);
			for(i=0; i<img.h; ++i)
				for(j=0; j<img.w; ++j)
					hist[outlqp[i*img.w+j]]++;

			/* to output the LQP
			Image out = lqptoimage(outlqp, img.w, img.h);

			strcpy(outname, "./outs/");
			strcat(outname, f->fts_name);
			if(!stbi_write_png(outname, out.w, out.h, comp, out.data, out.w*3))
				die("could not write %s", outname);

			free(out.data);
			out.data = NULL;
			*/

			free(img.data);
			img.data = NULL;
			free(outlqp);
			outlqp = NULL;

			processed++;
		}
		if(f->fts_info == FTS_DC)
			fprintf(stderr, "%s: cycle in directory tree", f->fts_path);
	}
	if(errno)
		die("fts_read errno is set");
	if(fts_close(tree) < 0)
		die("fts_close failed");
}

double
l2dist(uint32_t a, double *b)
{
	int i;
	double sum = 0.0;
	for(i=0; i<24; ++i)
		sum += pow((a>>i&1)-b[i], 2);
	return sqrt(sum);
}

double
drand(void)
{
	return ((double)rand()/(double)RAND_MAX);
}

int *
kmeans(int *data, long n, int k, double **centroids, double *err)
{
	int *labels = emalloc(n, sizeof(int));
	long h, i, j;
	long *counts = emalloc(k, sizeof(long)); 
	double olderr, newerr = DBL_MAX;
	double **c = centroids;
	double **c1 = emalloc(k, sizeof(double *));
	double mindist, dist;

	/* initialize */
	for(i=0; i<k; ++i) {
		c1[i] = emalloc(24, sizeof(double));
		for(j=0; j<24; ++j)
			c[i][j] = drand();
	}

	do {
		olderr = newerr, newerr = 0;
		/* clear old counts and temp centroids */
		for(i=0; i<k; counts[i++] = 0)
			for(j=0; j<24; c1[i][j++] = 0);

		/* for each datapoint, */
		for(h=0; h<n; ++h) {
			if(data[h]>10) {
				/* identify closest cluster */
				mindist = DBL_MAX;
				for(i=0; i<k; ++i) {
					dist = l2dist(h, c[i]);
					if(dist<mindist) {
						labels[h] = i;
						mindist = dist;
					}
				}
				/* update size and temp centroid of dest cluster */
				for(j=0; j<24; ++j)
					c1[labels[h]][j] += h>>j&1;
				counts[labels[h]]++;
				newerr += mindist;
			}
		}

		/* update all centroids */
		for(i=0; i<k; ++i)
			for(j=0; j<24; ++j)
				c[i][j] = counts[i] ? c1[i][j] / counts[i] : c1[i][j];

	} while(fabs(newerr-olderr)>0.01);

	for(i=0; i<k; ++i)
		free(c1[i]);
	free(c1);
	free(counts);
	*err = newerr;
	return labels;
}

void *
emalloc(size_t num, size_t size)
{
	void *mem;
	mem	= calloc(num, size);
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
