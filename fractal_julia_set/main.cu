#include <iostream>
#include "BitmapUtility.h"

const int DIM = 1024;

struct cuComplex {
	float r;
	float i;
	cuComplex( float a, float b ) : r(a), i(b) {}
	float magnitude2( void ) { return r * r + i * i; }
	cuComplex operator*(const cuComplex& a) {
		return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
	}
	cuComplex operator+(const cuComplex& a) {
		return cuComplex(r+a.r, i+a.i);
	}
};

int juliaCPU(int x, int y)
{
	const float scale = 1.5;
	float jx = scale * (float)(DIM/2 - x)/(DIM/2);
	float jy = scale * (float)(DIM/2 - y)/(DIM/2);
	cuComplex c(-0.8, 0.156);
	cuComplex a(jx, jy);
	int i = 0;
	for (i=0; i<200; i++) {
		a = a * a + c;
		if (a.magnitude2() > 1000)
			return 0;
	}
	return 1;
}

void runFractalRoutineOnCPU(unsigned char* bitmapPtr)
{
	for (int y=0; y<DIM; y++) {
		for (int x=0; x<DIM; x++) {
			int offset = x + y * DIM;
			int juliaValue = julia( x, y );
			bitmapPtr[offset*bytesPerPixel + 0] = 255 * juliaValue;
			bitmapPtr[offset*bytesPerPixel + 1] = 0;
			bitmapPtr[offset*bytesPerPixel + 2] = 0;
		}
	}
}

int main(){
    int height = DIM;
    int width = DIM;
    unsigned char image[height][width][bytesPerPixel];

    runFractalRoutineOnCPU((unsigned char*)image);

    generateBitmapImage((unsigned char*)image, height, width, "fractal.bmp");
}


