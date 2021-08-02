#include "BitmapWriter.h"

//max 16 classes
int COLOR_MAP_INDIA[16][3] = {{ 140, 67, 46 },{ 0, 0, 255 },{ 255, 100, 0 },{ 0, 255, 123 },{ 164, 75, 155 },{ 101, 174, 255 },{ 118, 254, 172 },{ 60, 91, 112 },{ 255, 255, 0 },{ 255, 255, 125 },{ 255, 0, 255 },{ 100, 0, 255 },{ 0, 172, 254 },{ 0, 255, 0 },{ 171, 175, 80 },{ 101, 193, 60 }};
int COLOR_MAP_UNI[16][3] = {{ 192, 192, 192 },{ 0, 255, 0 },{ 0, 255, 255 },{ 0, 128, 0 },{ 0, 255, 0 },{ 165, 82, 41 },{ 128, 0, 128 },{ 255, 0, 0 },{ 255, 255, 0 }};
int COLOR_MAP_CENTER[16][3] = {{ 0, 0, 255 },{ 0, 128, 0 },{ 0, 255, 0 },{ 255, 0, 0 },{ 142, 71, 2 },{ 192, 192, 192 },{ 0, 255, 255 },{ 246, 110, 0 },{ 255, 255, 0 }};
int COLOR_MAP_DC[16][3] = {{ 204, 102, 102 },{ 153, 51, 0 },{ 204, 153, 0 },{ 0, 255, 0 },{ 0, 102, 0 },{ 0, 51, 255 },{ 153, 153, 153 }};


void writeBMP(double *data, int w, int h, char const *filename, char const *type) {
	FILE *f;
	int x, y, r, g, b, idx, i, j;
	unsigned char *img = NULL;
	int filesize = 54 + 3 * w*h;  //w is your image width, h is image height, both int
	if (img)
		free(img);
	img = (unsigned char *)malloc(3 * w*h);
	memset(img, 0, sizeof(unsigned char)*3*w*h);

	int (*COLOR_MAP)[3];

	if(strcmp(type,"india")==0){
		COLOR_MAP = COLOR_MAP_INDIA;
	}else if(strcmp(type,"uni")==0){
		COLOR_MAP = COLOR_MAP_UNI;
	}else if(strcmp(type,"center")==0){
		COLOR_MAP = COLOR_MAP_CENTER;
	}else if(strcmp(type,"dc")==0){
		COLOR_MAP = COLOR_MAP_DC;
	}else{
		printf("It was not possible to find a correct color map.\n");
		exit(1);
	}

	for (i = 0; i<w; i++)
	{
		for (j = 0; j<h; j++)
		{
			x = i; y = (h - 1) - j;
			idx = (int)data[j + i*w];
			r = COLOR_MAP[idx][0];
			g = COLOR_MAP[idx][1];
			b = COLOR_MAP[idx][2];
			img[(x + y*w) * 3 + 2] = (unsigned char)(r);
			img[(x + y*w) * 3 + 1] = (unsigned char)(g);
			img[(x + y*w) * 3 + 0] = (unsigned char)(b);
		}
	}

	unsigned char bmpfileheader[14] = { 'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0 };
	unsigned char bmpinfoheader[40] = { 40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0 };
	unsigned char bmppad[3] = { 0,0,0 };

	bmpfileheader[2] = (unsigned char)(filesize);
	bmpfileheader[3] = (unsigned char)(filesize >> 8);
	bmpfileheader[4] = (unsigned char)(filesize >> 16);
	bmpfileheader[5] = (unsigned char)(filesize >> 24);

	bmpinfoheader[4] = (unsigned char)(w);
	bmpinfoheader[5] = (unsigned char)(w >> 8);
	bmpinfoheader[6] = (unsigned char)(w >> 16);
	bmpinfoheader[7] = (unsigned char)(w >> 24);
	bmpinfoheader[8] = (unsigned char)(h);
	bmpinfoheader[9] = (unsigned char)(h >> 8);
	bmpinfoheader[10] = (unsigned char)(h >> 16);
	bmpinfoheader[11] = (unsigned char)(h >> 24);

	f = fopen(filename, "wb");
	fwrite(bmpfileheader, 1, 14, f);
	fwrite(bmpinfoheader, 1, 40, f);
	//for (i = 0; i<h; i++)
	for (i = h - 1; i >= 0; i--)
	{
		fwrite(img + (w*(h - i - 1) * 3), 3, w, f);
		fwrite(bmppad, 1, (4 - (w * 3) % 4) % 4, f);
	}
	fclose(f);
}
