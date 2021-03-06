#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <unistd.h>
#include <math.h>
#include <ctype.h>

#define STB_IMAGE_IMPLEMENTATION
#include"../../../stb/stb_image.h" 
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include"../../../stb/stb_image_write.h" 


#define CL_TARGET_OPENCL_VERSION 120
#include "../ocl_boiler.h"


size_t gws_align_log;

short isNumber(const char* string){
	int i=0;
	if(string==NULL)return 0;
	while(string[i]!=0){
		if(!isdigit(string[i])){return 0;}
		i++;
	}
	return 1;
}

/*
float laplacianOfGaussian(float x, float y, float sigma){
	float num = ((x*x+y*y)-2*(sigma*sigma));
	float denom = 2*M_PI*(pow(sigma, 6));
	float expon = exp(-(x*x+y*y)/(2*(sigma*sigma)));
	return num*expon/denom;
}


void computeLogKernel(float * kernel_matrix, cl_int kwidth, cl_int kheight, float sigma){
	float sum_hg = 0, sum_k = 0;
	float curr_elem, hg;
	for(int i=0; i<kheight; ++i){
		for(int j=0; j<kwidth; ++j){
			float x = j-kwidth/2;
			float y = i-kheight/2;
			kernel_matrix[j*kheight+i] = laplacianOfGaussian(x,y,sigma);
		}
	}
}*/

void computeLogKernel(float * kernel_matrix, cl_int kwidth, cl_int kheight, float sigma){
	float sum_hg = 0, sum_k = 0;
	float curr_elem, hg;
	for(int i=0; i<kheight; ++i){
		for(int j=0; j<kwidth; ++j){
			float x = j-kwidth/2;
			float y = i-kheight/2;
			curr_elem = exp(-(x*x + y*y)/(2*(sigma*sigma)));
			sum_hg += curr_elem;
			kernel_matrix[j*kheight+i] = curr_elem*(x*x + y*y-2*(sigma*sigma))/(pow(sigma,4));
		}
	}
	
	for(int i=0; i<kheight; ++i){
		for(int j=0; j<kwidth; ++j){
			kernel_matrix[j*kheight+i] /= sum_hg;
			sum_k += kernel_matrix[j*kheight+i];
		}
	}

	for(int i=0; i<kheight; ++i){
		for(int j=0; j<kwidth; ++j){
			kernel_matrix[j*kheight+i] = kernel_matrix[j*kheight+i] - sum_k/(kwidth*kheight);
		}
	}
}

void print_matrix(float * matrix, int height, int width){
	for(int i=0; i<height; ++i){
		for(int j=0; j<width; ++j){
			printf("%f ", matrix[j*height+i]);
		}
		printf("\n");
	}
}

cl_event log_convolution(cl_kernel log_k, cl_command_queue que,
	cl_mem d_output, cl_mem d_input,
	cl_int nrows, cl_int ncols, cl_mem d_kernel_matrix, cl_int kwidth, cl_int kheight)
{
	const size_t gws[] = { round_mul_up(ncols, gws_align_log), nrows };
	cl_event log_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(log_k, i++, sizeof(d_output), &d_output);
	ocl_check(err, "set log arg %d", i-1);
	err = clSetKernelArg(log_k, i++, sizeof(d_input), &d_input);
	ocl_check(err, "set log arg %d", i-1);
	err = clSetKernelArg(log_k, i++, sizeof(d_kernel_matrix), &d_kernel_matrix);
	ocl_check(err, "set log arg %d", i-1);
	err = clSetKernelArg(log_k, i++, sizeof(kwidth), &kwidth);
	ocl_check(err, "set log arg %d", i-1);
	err = clSetKernelArg(log_k, i++, sizeof(kheight), &kheight);
	ocl_check(err, "set log arg %d", i-1);

	err = clEnqueueNDRangeKernel(que, log_k, 2,
		NULL, gws, NULL,
		0, NULL, &log_evt);

	ocl_check(err, "enqueue log");

	return log_evt;
}



void usage(int argc){
	if(argc<3){
		fprintf(stderr,"Usage: ./log <image.png> <output.png> [sigma]\n ");
		fprintf(stderr,"the image is an image with 4 channels (R,G,B,transparency)\n");
		fprintf(stderr,"sigma is the standard deviation to create the kernel for the LoG operator (default 0.5)\n");
		fprintf(stderr,"output is the name of the output image\n");

		exit(1);
	}
}

unsigned char* arrayOfMaxValuesUC(unsigned int dim){
	unsigned char* ret=malloc(sizeof(float)*dim);
	for(int i=0;i<dim;i++)
		ret[i]=0xff;
	return ret;
}

unsigned char* grayscale2RGBA(unsigned char* inputGray,int width, int height){
	unsigned int dimension = width*height;
	unsigned char* ret=malloc(sizeof(float)*dimension*4);
	unsigned char* maxValues = arrayOfMaxValuesUC(dimension);

	memcpy(ret,inputGray,dimension*sizeof(float));
	memcpy(ret+dimension*sizeof(float),inputGray,dimension*sizeof(float));
	memcpy(ret+2*dimension*sizeof(float),inputGray,dimension*sizeof(float));
	memcpy(ret+3*dimension*sizeof(float),maxValues,dimension*sizeof(float));

	free(maxValues);
	free(inputGray);
	return ret;
}

int main(int argc, char ** args){
	usage(argc);
	float sigma = 0.5;
	if(argc > 3) sigma = atof(args[3]);
	int width,height,channels;
	// caricamento immagine in memoria come array di unsigned char
	unsigned char * img = stbi_load(args[1],&width,&height,&channels,STBI_rgb_alpha);
	if(channels==1){
		printf("grayscale image with only one channel, doing transformation to 4 channels");
		img = grayscale2RGBA(img,width,height);
		channels=4;
	}

	if(img==NULL){
		printf("error while loading the image %s\n",args[1]);
		exit(1);
	}
	printf("image loaded with  %i width, %i height and %i channels\n",width,height,channels);
	if (channels < 3) {
                fprintf(stderr, "source image must have 4 channels (<RGB,alpha> or some other format with transparency and 3 channels for color space)\n");
                exit(1);
    }

	unsigned char * outimg = NULL;
	int data_size=width*height*channels;
	int dstwidth=width,dstheight=height;
	int dstdata_size=dstwidth*dstheight*channels;
	cl_platform_id p = select_platform();
	cl_device_id d = select_device(p);
	cl_context ctx = create_context(p, d);
	cl_command_queue que = create_queue(ctx, d);
	cl_program prog = create_program("segmentation.ocl", ctx, d);
	int err=0;

	cl_kernel log_k = clCreateKernel(prog, "convolution", &err);
	ocl_check(err, "create kernel convolution");
    
	/* get information about the preferred work-group size multiple */
	err = clGetKernelWorkGroupInfo(log_k, d,
		CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
		sizeof(gws_align_log), &gws_align_log, NULL);
	ocl_check(err, "preferred wg multiple for log");

	cl_mem d_kernel_matrix = NULL, d_input = NULL, d_output = NULL;

	const cl_image_format fmt = {
		.image_channel_order = CL_RGBA,
        //.image_channel_data_type = CL_FLOAT,
		.image_channel_data_type = CL_UNSIGNED_INT8,
	};
	const cl_image_desc desc = {
		.image_type = CL_MEM_OBJECT_IMAGE2D,
		.image_width = width,
		.image_height = height,
		//.image_row_pitch = src.data_size/src.height,
	};

	d_input = clCreateImage(ctx,
		CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
		&fmt, &desc,
		img,
		&err);
	ocl_check(err, "create image d_input");

	//LoG matrix 5x5
	float kernel_matrix[] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0};
	/*
	float kernel_matrix[] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0,
		};
	*/
	cl_int kwidth = 5, kheight = 5;
	computeLogKernel(kernel_matrix, kwidth, kheight, sigma);
	print_matrix(kernel_matrix, kheight, kwidth);

	d_kernel_matrix = clCreateBuffer(ctx,
	CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	sizeof(kernel_matrix), kernel_matrix,
		&err);
	ocl_check(err, "create buffer d_kernel_matrix");

	d_output = clCreateBuffer(ctx,
	CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
	dstdata_size, NULL,
		&err);
	ocl_check(err, "create buffer d_output");

	cl_event log_evt, map_evt;

	log_evt = log_convolution(log_k, que, d_output, d_input, height, width, d_kernel_matrix, kwidth, kheight);

	outimg = clEnqueueMapBuffer(que, d_output, CL_TRUE,
		CL_MAP_READ,
		0, dstdata_size,
		1, &log_evt, &map_evt, &err);
	ocl_check(err, "enqueue map d_output");

	const double runtime_log_ms = runtime_ms(log_evt);
	const double runtime_map_ms = runtime_ms(map_evt);
	const double total_runtime_ms = runtime_log_ms+runtime_map_ms;

	const double log_bw_gbs = dstdata_size/1.0e6/runtime_log_ms;
	const double map_bw_gbs = dstdata_size/1.0e6/runtime_map_ms;

	printf("log convolution: %dx%d int in %gms: %g GB/s %g GE/s\n",
		height, width, runtime_log_ms, log_bw_gbs, height*width/1.0e6/runtime_log_ms);
	printf("map: %dx%d int in %gms: %g GB/s %g GE/s\n",
		dstheight, dstwidth, runtime_map_ms, map_bw_gbs, dstheight*dstwidth/1.0e6/runtime_map_ms);
	printf("Total runtime: %g ms\n", total_runtime_ms);

	char outputImage[128];
	sprintf(outputImage,"%s",args[2]);
	printf("image saved as %s\n",outputImage);
	stbi_write_png(outputImage,dstwidth,dstheight,channels,outimg,channels*dstwidth);

	err = clEnqueueUnmapMemObject(que, d_output, outimg,
		0, NULL, NULL);
	ocl_check(err, "unmap output");

	clReleaseMemObject(d_output);
	clReleaseMemObject(d_input);
	clReleaseMemObject(d_kernel_matrix);
	clReleaseKernel(log_k);
	clReleaseProgram(prog);
	clReleaseCommandQueue(que);
	clReleaseContext(ctx);

	stbi_image_free(img);
	return 0;
}
