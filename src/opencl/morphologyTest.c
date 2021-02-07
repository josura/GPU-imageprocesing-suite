#include "morphology.h"


void usage(int argc){
	if(argc<4){
		fprintf(stderr,"Usage: ./erosion <image.png> <strel.png> <output.png>\n ");
		fprintf(stderr,"the image is an image with 4 channels (R,G,B,transparency)\n");
		fprintf(stderr,"strel is an image that is used as a structuring element\n");
		fprintf(stderr,"output is the name of the output image\n");

		exit(1);
	}
}

int main(int argc, char ** args){
	usage(argc);
	int width,height,channels,strelwidth,strelheight,strelchannels;
	// caricamento immagine in memoria come array di unsigned char
	unsigned char * img= stbi_load(args[1],&width,&height,&channels,STBI_rgb_alpha);

	unsigned char * imgstrel= stbi_load(args[2],&strelwidth,&strelheight,&strelchannels,STBI_rgb_alpha);

    unsigned char * processedErosion = fullErosion(img,imgstrel,width,height,channels,strelwidth,strelheight,strelchannels);
    if(processedErosion==NULL){
        fprintf(stderr,"problems in method for erosion\n");
        exit(1);
    }
    

	stbi_write_png(args[3],width,height,channels,processedErosion,channels*width);
    printf("image saved as %s\n",args[3]);

    stbi_image_free(img);
    stbi_image_free(imgstrel);
    stbi_image_free(processedErosion);

}