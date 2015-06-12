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

#define efree(p)\
do {\
	if(!p)\
		fprintf(stderr, "tried to free null pointer at line %d\n", __LINE__);\
	else\
		free(p);\
	p = NULL;\
} while(0)\

int nfilestrained = 10;
int nkmeans = 10;
int K = 150;
int grid = 10;

struct Image {
	unsigned char *data;
	int w, h;
};
typedef struct Image Image;

struct Pathnode {
	char path[80];
	char name[80];
	uint32_t *lqp;
	struct Pathnode *next;
};
typedef struct Pathnode Pathnode;

uint32_t *lqp(Image);
void printbinary(int, void *);
Image lqptoimage(uint32_t *, int w, int h);
void tograyscale(Image);
unsigned char *pixelat(Image, int x, int y);
int entcmp(const FTSENT **, const FTSENT **);
Pathnode *getfiles(char *dir, char *pattern);
void freepaths(Pathnode *);
int *trainonfiles(int n, Pathnode *);
double kmeans(int *data, long n, int k, double ***centroids);
double l2dist(uint32_t, double *);
void *emalloc(size_t num, size_t size);
void die(char *fmt, ...);

/* training:
 * go through every picture, calculate a LQP, add to histogram
 * build list of binary patterns by filtering by threshold
 * do k-means clustering (probably multiple times, choose least err)
 * assign every possible binary number to its nearest neighbor,
 * create look up table
 *
 * to find similar faces, build histogram of code points
 * sort by cosine similarity on histogram
 */

int 
main(void)
{
	long i, j;
	double minerr;
	int minindex;

	srand(time(NULL));
 
	double **trialcentroids[nkmeans];
	double trialerrs[nkmeans];

	Pathnode *files = getfiles("./orl_faces", "*.jpg");
	int *hist = trainonfiles(nfilestrained, files);

	long sum = 0;
	for(i=0; i<16777216; ++i)
		if(hist[i]) 
			sum++;
	printf("%ld patterns\n", sum);

	#pragma omp parallel for
	for(i=0; i<nkmeans; ++i) {
		trialerrs[i] = kmeans(hist, 16777216, K, &trialcentroids[i]);
		printf("finished k-means %ld\n", i);
		fflush(stdout);
	}

	minerr = DBL_MAX;
	minindex = 0;
	for(i=0; i<nkmeans; ++i) {
		if(trialerrs[i]<minerr) {
			minerr = trialerrs[i];
			minindex = i;
		}
	}
	double **centroids = trialcentroids[minindex];
	double err = trialerrs[minindex];

	printf("best k-means is %d, with error %f\n", minindex, err);

	Image img;
	int comp;
	if(!(img.data = stbi_load("1.jpg", &img.w, &img.h, &comp, 0)))
		die("could not load 1.jpg");
	if(comp != 1)
		die("picture has %d components, should be 1", comp);

	uint32_t *outlqp = lqp(img);

	Image outimg = lqptoimage(outlqp, img.w, img.h);
	if(!stbi_write_png("1lqp.png", img.w, img.h, comp, outimg.data, img.w))
		die("could not write patterns.png");
	efree(outimg.data);

	printf("building lookup table\n");
	int *lut = emalloc(16777216, sizeof(int));
	int mindistindex = 0;
	uint32_t ii;
	for(ii=0; ii<16777216; ++ii) {
		minerr = DBL_MAX;
		for(j=0; j<K; ++j) {
			double d = l2dist(ii, centroids[j]);
			if(d<minerr) {
				minerr = d;
				mindistindex = j;
			}
		}
		lut[ii] = mindistindex;
	}

	int x, y;
	uint32_t p;
	for(y=0; y<img.h; ++y) {
		for(x=0; x<img.w; ++x) {
			p = outlqp[y*img.w+x];
			*pixelat(img, x, y) = lut[p] * 255.0 / K;
		}
	}

	if(!stbi_write_png("patterns.png", img.w, img.h, comp, img.data, img.w))
		die("could not write patterns.png");

	efree(img.data);
	efree(outlqp);
	free(lut);

	efree(hist);
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
			v = data[i*w+j];
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
	return img.data+y*img.w+x;
}

int 
entcmp(const FTSENT **a, const FTSENT **b)
{
	return strcmp((*a)->fts_name, (*b)->fts_name);
}

int *
trainonfiles(int n, Pathnode *pn)
{
	/* 2^24 = 16777216 */
	int *hist	= emalloc(16777216, sizeof(int));

	int i, j;
	int comp;
	Image img;

	int processed = 0;
	while(pn && processed < n) {
		if(!(img.data = stbi_load(pn->path, &img.w, &img.h, &comp, 0)))
			die("could not load %s", pn->path);
		if(comp != 1)
			die("picture has %d components, should be 1", comp);

		if(!pn->lqp)
			pn->lqp = lqp(img);

		for(i=0; i<img.h; ++i)
			for(j=0; j<img.w; ++j)
				hist[pn->lqp[i*img.w+j]]++;

		efree(img.data);

		processed++;
		pn = pn->next;
	}

	return hist;
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

Pathnode *
getfiles(char *dir, char *pattern)
{
	FTS *tree;
	FTSENT *f;
	char *argv[] = { dir, NULL };

	Pathnode *head = NULL;
	Pathnode *curr = NULL;

	tree = fts_open(argv, FTS_LOGICAL | FTS_NOSTAT, entcmp);
	if(!tree)
		die("fts_open failed");
	while((f = fts_read(tree))) {
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
			curr = emalloc(1, sizeof(Pathnode));
			strlcpy(curr->path, f->fts_path, sizeof(curr->path));
			strlcpy(curr->name, f->fts_name, sizeof(curr->name));
			curr->lqp = NULL;
			curr->next = head;
			head = curr;
		}
		if(f->fts_info == FTS_DC)
			fprintf(stderr, "%s: cycle in directory tree", f->fts_path);
	}
	if(errno)
		die("fts_read errno is set");
	if(fts_close(tree) < 0)
		die("fts_close failed");
	return head;
}

void 
freepaths(Pathnode *p)
{
	Pathnode *q;
	while((q = p)) {
		p = p->next;
		if(q->lqp)
			efree(q->lqp);
		efree(q);
	}
}

double
kmeans(int *data, long n, int k, double ***centroids)
{
	int *labels = emalloc(n, sizeof(int));
	long h, i, j;
	long *counts = emalloc(k, sizeof(long)); 
	double olderr, err = DBL_MAX;
	double mindist, dist;

	*centroids = emalloc(k, sizeof(double *));
	double **c = *centroids;
	double **c1;

	/* initialize */
	c1 = emalloc(k, sizeof(double *));
	for(i=0; i<k; ++i) {
		c[i] = emalloc(24, sizeof(double));
		c1[i] = emalloc(24, sizeof(double));
		for(j=0; j<24; ++j)
			c[i][j] = drand();
	}


	do {
		olderr = err, err = 0;
		/* clear old counts and temp centroids */
		for(i=0; i<k; counts[i++] = 0)
			for(j=0; j<24; c1[i][j++] = 0);

		/* for each datapoint, */
		for(h=0; h<n; ++h) {
			if(data[h]) {
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
				err += mindist;
			}
		}

		/* update all centroids */
		for(i=0; i<k; ++i)
			for(j=0; j<24; ++j)
				c[i][j] = counts[i] ? c1[i][j] / counts[i] : c1[i][j];

	} while(fabs(err-olderr)>0.01);

	for(i=0; i<k; ++i)
		efree(c1[i]);
	efree(c1);
	efree(counts);
	return err;
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
