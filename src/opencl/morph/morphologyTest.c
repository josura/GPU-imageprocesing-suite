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


	//EROSION TEST
    unsigned char * processedErosion = fullErosion(img,imgstrel,width,height,channels,strelwidth,strelheight,strelchannels);
    if(processedErosion==NULL){
        fprintf(stderr,"problems in method for erosion\n");
        exit(1);
    }

	//DILATION TEST
	unsigned char * processedDilation = fullDilation(img,imgstrel,width,height,channels,strelwidth,strelheight,strelchannels);
    if(processedDilation==NULL){
        fprintf(stderr,"problems in method for dilation\n");
        exit(1);
    }

	//GRADIENT TEST
	unsigned char * processedGradient = fullGradient(img,imgstrel,width,height,channels,strelwidth,strelheight,strelchannels);
    if(processedGradient==NULL){
        fprintf(stderr,"problems in method for gradient\n");
        exit(1);
    }

	//CLOSING TEST
	unsigned char * processedClosing = fullClosing(img,imgstrel,width,height,channels,strelwidth,strelheight,strelchannels);
    if(processedClosing==NULL){
        fprintf(stderr,"problems in method for closing\n");
        exit(1);
    }

	//OPENING TEST
	unsigned char * processedOpening = fullOpening(img,imgstrel,width,height,channels,strelwidth,strelheight,strelchannels);
    if(processedOpening==NULL){
        fprintf(stderr,"problems in method for opening\n");
        exit(1);
    }

    //TOPHAT TEST
	unsigned char * processedTophat = fullTophat(img,imgstrel,width,height,channels,strelwidth,strelheight,strelchannels);
    if(processedTophat==NULL){
        fprintf(stderr,"problems in method for tophat\n");
        exit(1);
    }

    //BOTTOMHAT TEST
	unsigned char * processedBottomhat = fullBottomhat(img,imgstrel,width,height,channels,strelwidth,strelheight,strelchannels);
    if(processedBottomhat==NULL){
        fprintf(stderr,"problems in method for bottomhat\n");
        exit(1);
    }

    //HIT OR MISS TEST
	unsigned char * processedHM = fullHitorMiss(img,imgstrel,width,height,channels,strelwidth,strelheight,strelchannels);
    if(processedBottomhat==NULL){
        fprintf(stderr,"problems in method for bottomhat\n");
        exit(1);
    }

    
	char buffer[128];
	sprintf(buffer,"erosion%s",args[3]);
	stbi_write_png(buffer,width,height,channels,processedErosion,channels*width);
    printf("erosion image saved as %s\n",buffer);

	sprintf(buffer,"dilation%s",args[3]);
	stbi_write_png(buffer,width,height,channels,processedDilation,channels*width);
    printf("dilation image saved as %s\n",buffer);

	sprintf(buffer,"gradient%s",args[3]);
	stbi_write_png(buffer,width,height,channels,processedGradient,channels*width);
    printf("gradient image saved as %s\n",buffer);

	sprintf(buffer,"opening%s",args[3]);
	stbi_write_png(buffer,width,height,channels,processedOpening,channels*width);
    printf("opening image saved as %s\n",buffer);

	sprintf(buffer,"Closing%s",args[3]);
	stbi_write_png(buffer,width,height,channels,processedClosing,channels*width);
    printf("closing image saved as %s\n",buffer);

    sprintf(buffer,"tophat%s",args[3]);
	stbi_write_png(buffer,width,height,channels,processedTophat,channels*width);
    printf("tophat image saved as %s\n",buffer);

    sprintf(buffer,"bottomhat%s",args[3]);
	stbi_write_png(buffer,width,height,channels,processedBottomhat,channels*width);
    printf("bottomhat image saved as %s\n",buffer);

    sprintf(buffer,"hitormidd%s",args[3]);
	stbi_write_png(buffer,width,height,channels,processedHM,channels*width);
    printf("hit or miss image saved as %s\n",buffer);

    stbi_image_free(img);
    stbi_image_free(imgstrel);

    //TODO freeing memory for every uchar array but it is useless because the program has finished
    /*stbi_image_free(processedErosion);
    stbi_image_free(processedDilation);
    stbi_image_free(processedGradient);
    stbi_image_free(processedOpening);
    stbi_image_free(processedClosing);*/

}

