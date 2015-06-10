#include "stdio.h"
#include "math.h"

#define STB_IMAGE_IMPLEMENTATION
#include "inc/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "inc/stb_image_write.h"

int main(int argc, char *argv[]) 
{
	#pragma omp parallel for
	for(int n=0; n<2; ++n)
		printf("%d\n", n);

	int w, h, n;
	unsigned char *data = stbi_load("stannis.jpg", &w, &h, &n, 0);
	printf("loaded %d scanlines of %d pixels, with %d components\n", h, w, n);

	unsigned char *out = malloc(sizeof(unsigned char)*3*w*h);
	memset(out, 0, sizeof(unsigned char)*3*w*h);

	// for every pixel,
	for(int i=1; i<h-1; ++i) {
		for(int j=1; j<w-1; ++j) {
			int px = 3*(i*w+j);

			// for every neighbor,
			for(int y=-1; y<2; ++y) {
				for(int x=-1; x<2; ++x) {
					if(!(x == 0 && y == 0)) {
						int neighbor = 3*((i+y)*w+(j+x));
						unsigned char add = (data[neighbor] > data[px]) << ((y+1)*3+(x+1));
						out[px] += add;
						out[px+1] += add;
						out[px+2] += add;
						//printf("is now %d", out[px]);
					}
				}
			}

		}
	}

	stbi_write_png("stannis.png", w, h, n, out, w*3);

	return 0;
}


