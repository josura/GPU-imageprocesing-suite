#include "morphology.h"

void usage(int argc){
	if(argc<5){
		fprintf(stderr,"Usage: ./erosion <image.png> <strel.png> <output.png> <operation>\n ");
		fprintf(stderr,"the image is an image with 4 channels (R,G,B,transparency)\n");
		fprintf(stderr,"strel is an image that is used as a structuring element\n");
		fprintf(stderr,"output is the name of the output image\n");
		fprintf(stderr,"operation is the morphological operation\n");

		exit(1);
	}
}




int main(int argc, char ** args){
	usage(argc);
	int width,height,channels;
	unsigned char* processed =morphOperation(args[1],args[2],args[4],&width,&height,&channels);


	if(processed){
		char buffer[128];
		sprintf(buffer,"%s",args[3]);
		stbi_write_png(buffer,width,height,channels,processed,channels*width);
	} else {
		printf("error processing the image\n");
	}
	return 0;
}
