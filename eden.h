int nfilestrained = 2;
int nkmeans = 2;
int K = 2;

#define efree(p)\
do {\
	if(!p)\
		fprintf(stderr, "tried to free null pointer at line %d\n", __LINE__);\
	else\
		free(p);\
	p = NULL;\
} while(0)\

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
	float *hist;
	int histlen;
	struct Pathnode *next;
} Pathnode;

Pattern lqp(Image);
float *patternhistogram(Pattern, int *lut, int k, int *len);
void printbinary(int, void *);
Image lqptoimage(Pattern);
uint32_t *histtoarr(int *hist, long *elts);
void tograyscale(Image);
unsigned char *pixelat(Image, int x, int y);
int entcmp(const FTSENT **, const FTSENT **);
Pathnode *getfiles(char *dir, char *pattern);
void freepaths(Pathnode *);
uint32_t *getpatterns(int n, Pathnode *, long *elts);
float kmeans(uint32_t *data, long n, int k, float ***centroids);
void freecentroids(float ***centroids, long n, int k);
float l2dist(uint32_t, float *);
float otherl2dist(uint32_t, float *);
float alsol2dist(float *a, float *b);
float cosdist(float *, float *, int len);
void *emalloc(size_t num, size_t size);
void die(char *fmt, ...);

