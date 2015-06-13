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

typedef struct Image {
	unsigned char *data;
	int w, h;
} Image;

typedef struct Pattern {
	uint32_t *data;
	int w, h;
} Pattern;

typedef struct Pathnode {
	char path[80];
	char name[80];
	Pattern pattern;
	struct Pathnode *next;
} Pathnode;

Pattern lqp(Image);
void printbinary(int, void *);
Image lqptoimage(Pattern);
uint32_t *histtoarr(int *hist, long *elts);
void tograyscale(Image);
unsigned char *pixelat(Image, int x, int y);
int entcmp(const FTSENT **, const FTSENT **);
Pathnode *getfiles(char *dir, char *pattern);
void freepaths(Pathnode *);
uint32_t *trainonfiles(int n, Pathnode *, long *elts);
double kmeans(uint32_t *data, long n, int k, double ***centroids);
void freecentroids(double ***centroids, long n, int k);
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
	long i;

	double **trialcentroids[nkmeans];
	double trialerrs[nkmeans];

	double err = DBL_MAX;
	double **centroids;
	int min;

	srand(time(NULL));

	Pathnode *files = getfiles("./orl_faces", "*.jpg");

	/* build an array of all patterns */
	long elts;
	uint32_t *patterns = trainonfiles(nfilestrained, files, &elts);
	printf("%ld patterns\n", elts);

	/* find k-means clusters */
	#pragma omp parallel for
	for(i=0; i<nkmeans; ++i) {
		trialerrs[i] = kmeans(patterns, elts, K, &trialcentroids[i]);
		printf("finished k-means %ld\n", i);
		fflush(stdout);
	}

	/* find the most accurate k-means clusters */
	err = DBL_MAX;
	min = 0;
	for(i=0; i<nkmeans; ++i)
		if(trialerrs[i]<trialerrs[min])
			min = i;

	err = trialerrs[min];
	centroids = trialcentroids[min];
	printf("best k-means has error %f\n", err);

	/* load the test image */
	Image img;
	int comp;
	if(!(img.data = stbi_load("1.jpg", &img.w, &img.h, &comp, 0)))
		die("could not load 1.jpg");
	if(comp != 1)
		die("picture has %d components, should be 1", comp);

	Pattern outlqp = lqp(img);
	Image outimg = lqptoimage(outlqp);
	if(!stbi_write_png("1lqp.png", img.w, img.h, comp, outimg.data, img.w))
		die("could not write patterns.png");

	printf("building lookup table\n");
	int *lut = emalloc(1L<<24, sizeof(*lut));
	uint32_t ii;
	#pragma omp parallel for
	for(ii=0; ii<(1L<<24); ++ii) {
		double minerr = DBL_MAX;
		int mindistindex = 0;
		double d;
		int j;
		for(j=0; j<K; ++j) {
			d = l2dist(ii, centroids[j]);
			if(d<minerr) {
				minerr = d;
				mindistindex = j;
			}
		}
		lut[ii] = mindistindex;
	}

	printf("generating image\n");
	int x, y;
	uint32_t p;
	for(y=0; y<img.h; ++y) {
		for(x=0; x<img.w; ++x) {
			p = outlqp.data[y*img.w+x];
			*pixelat(img, x, y) = lut[p] * 255.0 / K;
		}
	}
	if(!stbi_write_png("patterns.png", img.w, img.h, comp, img.data, img.w))
		die("could not write patterns.png");

	freepaths(files);
	freecentroids(trialcentroids, nkmeans, K);

	efree(patterns);
	efree(outimg.data);
	efree(outlqp.data);
	efree(img.data);
	efree(lut);

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

Pattern
lqp(Image img)
{
	unsigned int radius, n;
	unsigned int i, j;
	double x, y, tx, ty;
	int fx, fy, cx, cy;
	double w1, w2, w3, w4;
	double t;
	uint32_t add;

	Pattern p;
	p.w = img.w;
	p.h = img.h;
	p.data = emalloc(img.w*img.h, sizeof(*p.data));

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
					p.data[(i-radius)*img.w+(j-radius)] += add;
				}
			}
		}
	}
	return p;
}

Image 
lqptoimage(Pattern p)
{
	int i, j;
	unsigned char *px, v;
	Image out;
	out.w = p.w;
	out.h = p.h;
	out.data = emalloc(p.w*p.h*3, sizeof(*out.data));

	for(i=0; i<p.h; ++i)
		for(j=0; j<p.w; ++j){
			px = pixelat(out, j, i);
			v = p.data[i*p.w+j];
			px[0] = v;
			px[1] = v;
			px[2] = v;
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

uint32_t *
trainonfiles(int n, Pathnode *pn, long *elts)
{
	int *hist	= emalloc(1L<<24, sizeof(*hist));

	int i, j;
	int comp;
	Image img;

	int processed = 0;
	while(pn && processed < n) {
		if(!pn->pattern.data) {
			if(!(img.data = stbi_load(pn->path, &img.w, &img.h, &comp, 0)))
				die("could not load %s", pn->path);
			if(comp != 1)
				die("picture has %d components, should be 1", comp);
			pn->pattern = lqp(img);
			efree(img.data);
		}

		for(i=0; i<img.h; ++i)
			for(j=0; j<img.w; ++j)
				hist[pn->pattern.data[i*img.w+j]]++;

		processed++;
		pn = pn->next;
	}

	uint32_t *arr = histtoarr(hist, elts);
	efree(hist);
	return arr;
}

uint32_t *
histtoarr(int *hist, long *elts)
{
	size_t size = 128;
	long ii;
	uint32_t *arr = emalloc(size, sizeof(*arr));

	*elts = 0;
	for(ii=0; ii<(1L<<24); ++ii) {
		if(hist[ii]) {
			if(*elts>=size) {
				size *= 2;
				arr = realloc(arr, size*sizeof(int));
			}
			arr[*elts] = ii;
			(*elts)++;
		}
	}
	return arr;
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
			curr = emalloc(1, sizeof(*curr));
			strlcpy(curr->path, f->fts_path, sizeof(curr->path));
			strlcpy(curr->name, f->fts_name, sizeof(curr->name));
			curr->pattern.data = NULL;
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
		if(q->pattern.data)
			efree(q->pattern.data);
		efree(q);
	}
}

double
kmeans(uint32_t *data, long n, int k, double ***centroids)
{
	int *labels = emalloc(n, sizeof(*labels));
	long h, i, j;
	long *counts = emalloc(k, sizeof(*counts)); 
	double olderr, err = DBL_MAX;
	double mindist, dist;

	*centroids = emalloc(k, sizeof(**centroids));
	double **c = *centroids;
	double **c1;

	c1 = emalloc(k, sizeof(*c1));
	for(i=0; i<k; ++i) {
		c[i] = emalloc(24, sizeof(*c[i]));
		c1[i] = emalloc(24, sizeof(*c1[i]));
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
			/* identify closest cluster */
			mindist = DBL_MAX;
			for(i=0; i<k; ++i) {
				dist = l2dist(data[h], c[i]);
				if(dist<mindist) {
					labels[h] = i;
					mindist = dist;
				}
			}
			/* update size and temp centroid of dest cluster */
			for(j=0; j<24; ++j)
				c1[labels[h]][j] += data[h]>>j&1;
			counts[labels[h]]++;
			err += mindist;
		}

		/* update all centroids */
		for(i=0; i<k; ++i)
			for(j=0; j<24; ++j)
				c[i][j] = counts[i] ? c1[i][j] / counts[i] : c1[i][j];

	} while(fabs(err-olderr)>0.001);

	for(i=0; i<k; ++i)
		efree(c1[i]);
	efree(c1);
	efree(counts);
	efree(labels);
	return err;
}
void 
freecentroids(double ***centroids, long n, int k)
{
	long i, j;
	for(i=0; i<n; ++i) {
		for(j=0; j<k; ++j)
			efree(centroids[i][j]);
		efree(centroids[i]);
	}
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
