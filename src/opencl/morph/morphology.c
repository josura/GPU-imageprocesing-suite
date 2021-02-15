#include "morphology.h"

void usage(int argc){
	if(argc<5){
		fprintf(stderr,"Usage: ./erosion <image.png> <strel.png> <output.png> <operation> [mask.png] [#iterations]\n ");
		fprintf(stderr,"the image is an image with 4 channels (R,G,B,transparency)\n");
		fprintf(stderr,"strel is an image that is used as a structuring element\n");
		fprintf(stderr,"output is the name of the output image\n");
		fprintf(stderr,"operation is the morphological operation\n");
		fprintf(stderr," the mask is an image used for geodesic reconstruction and it is optional\n");
		fprintf(stderr," #iterations is the number of recursive calls used for geodesic reconstruction and it is optional\n");

		exit(1);
	}
}




int main(int argc, char ** args){
	usage(argc);
	int width,height,channels,numIteration=0;
	unsigned char* processed=NULL;
	if(argc<6 && !strstr(args[4],"geodesic")){
		processed =morphOperation(args[1],args[2],args[4],&width,&height,&channels,NULL,numIteration);
	}
	else {
		if(argc>6 && isNumber(args[6])){numIteration=atoi(args[6]);}
		printf("morph geodesic with %i numiterations and %s mask\n ",numIteration,args[5]);
		processed =morphOperation(args[1],args[2],args[4],&width,&height,&channels,args[5],numIteration);
	}


	if(processed){
		char buffer[128];
		sprintf(buffer,"%s",args[3]);
		stbi_write_png(buffer,width,height,channels,processed,channels*width);
	} else {
		printf("error processing the image\n");
	}
	return 0;
}
