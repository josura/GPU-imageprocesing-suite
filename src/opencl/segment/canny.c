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


size_t gws_align_drog;

short isNumber(const char* string){
	int i=0;
	if(string==NULL)return 0;
	while(string[i]!=0){
		if(!isdigit(string[i])){return 0;}
		i++;
	}
	return 1;
}

void computeDrogKernel(float * kernel_matrix, cl_int kwidth, cl_int kheight, float sigma){
	float sum_k = 0;
	float curr_elem;
	for(int i=0; i<kheight; ++i){
		for(int j=0; j<kwidth; ++j){
			float x = j-kwidth/2;
			float y = i-kheight/2;
			curr_elem = -x/(sigma*sigma) * (exp(-(x*x+y*y)/(2*sigma*sigma)));
			kernel_matrix[j*kheight+i] = curr_elem;
		}
	}
	
	for(int i=0; i<kwidth; ++i){
		for(int j=0; j<kheight; ++j){
			if(i != kwidth/2) kernel_matrix[j*kheight+i] /= 2;
			//kernel_matrix[j*kheight+i] = kernel_matrix[j*kheight+i] - sum_k/(kwidth*kheight);
		}
	}
}

void print_matrix(float * matrix, int height, int width){
	for(int i=0; i<height; ++i){
		for(int j=0; j<width; ++j){
			printf("%f ", matrix[i*width+j]);
		}
		printf("\n");
	}
}

cl_event drog_convolution(cl_kernel drog_k, cl_command_queue que,
	cl_mem d_magnitudes, cl_mem d_input, cl_mem d_angles,
	cl_int nrows, cl_int ncols, cl_mem d_kernel_matrix, cl_int kwidth, cl_int kheight)
{
	const size_t gws[] = { round_mul_up(ncols, gws_align_drog), nrows };
	cl_event drog_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(drog_k, i++, sizeof(d_magnitudes), &d_magnitudes);
	ocl_check(err, "_set drog arg %d", i-1);
	err = clSetKernelArg(drog_k, i++, sizeof(d_angles), &d_angles);
	ocl_check(err, "_set drog arg %d", i-1);
	err = clSetKernelArg(drog_k, i++, sizeof(d_input), &d_input);
	ocl_check(err, "_set drog arg %d", i-1);
	err = clSetKernelArg(drog_k, i++, sizeof(d_kernel_matrix), &d_kernel_matrix);
	ocl_check(err, "_set drog arg %d", i-1);
	err = clSetKernelArg(drog_k, i++, sizeof(kwidth), &kwidth);
	ocl_check(err, "_set drog arg %d", i-1);
	err = clSetKernelArg(drog_k, i++, sizeof(kheight), &kheight);
	ocl_check(err, "_set drog arg %d", i-1);

	err = clEnqueueNDRangeKernel(que, drog_k, 2,
		NULL, gws, NULL,
		0, NULL, &drog_evt);

	ocl_check(err, "enqueue drog");

	return drog_evt;
}

cl_event non_maxima_suppression(cl_kernel suppress_k, cl_command_queue que,
	cl_mem d_magnitudes, cl_mem d_supp_magnitudes, cl_mem d_angles,
	cl_int nrows, cl_int ncols)
{
	const size_t gws[] = { round_mul_up(ncols, gws_align_drog), nrows };
	cl_event suppress_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(suppress_k, i++, sizeof(d_magnitudes), &d_magnitudes);
	ocl_check(err, "_set suppress arg %d", i-1);
	err = clSetKernelArg(suppress_k, i++, sizeof(d_supp_magnitudes), &d_supp_magnitudes);
	ocl_check(err, "_set suppress arg %d", i-1);
	err = clSetKernelArg(suppress_k, i++, sizeof(d_angles), &d_angles);
	ocl_check(err, "_set suppress arg %d", i-1);

	err = clEnqueueNDRangeKernel(que, suppress_k, 2,
		NULL, gws, NULL,
		0, NULL, &suppress_evt);

	ocl_check(err, "enqueue suppress");

	return suppress_evt;
}

cl_event m_hysteresis(cl_kernel hysteresis_k, cl_command_queue que,
	cl_mem d_output, cl_mem d_magnitudes, cl_uint low_threshold, cl_uint high_threshold,
	cl_int nrows, cl_int ncols)
{
	const size_t gws[] = { round_mul_up(ncols, gws_align_drog), nrows };
	cl_event hysteresis_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(hysteresis_k, i++, sizeof(d_output), &d_output);
	ocl_check(err, "_set hysteresis arg %d", i-1);
	err = clSetKernelArg(hysteresis_k, i++, sizeof(d_magnitudes), &d_magnitudes);
	ocl_check(err, "_set hysteresis arg %d", i-1);
	err = clSetKernelArg(hysteresis_k, i++, sizeof(low_threshold), &low_threshold);
	ocl_check(err, "_set hysteresis arg %d", i-1);
	err = clSetKernelArg(hysteresis_k, i++, sizeof(high_threshold), &high_threshold);
	ocl_check(err, "_set hysteresis arg %d", i-1);

	err = clEnqueueNDRangeKernel(que, hysteresis_k, 2,
		NULL, gws, NULL,
		0, NULL, &hysteresis_evt);

	ocl_check(err, "enqueue hysteresis");

	return hysteresis_evt;
}

void usage(int argc){
	if(argc<3){
		fprintf(stderr,"Usage: ./canny <image.png> <output.png> [sigma] [low_threshold] [high_threshold]\n ");
		fprintf(stderr,"The image needs to have 4 channels (R,G,B,transparency)\n");
		fprintf(stderr,"It will be converted to grayscale with 1 channel\n");
		fprintf(stderr,"sigma is the standard deviation for the DroG kernel (default 2.5)\n");
		fprintf(stderr,"low_threshold and high_threshold are the two threshold for hysteresis\n");
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

unsigned char* RGBA2grayscale(unsigned char* inputRGB, int width, int height){
	unsigned int dimension = width*height;
	unsigned char* ret=malloc(sizeof(unsigned char)*dimension*4);
	unsigned char r, g, b, alpha, gray;

	for(int i=0; i<dimension; ++i){
		r = inputRGB[i<<2];
		g = inputRGB[(i<<2)+1];
		b = inputRGB[(i<<2)+2];
		alpha = inputRGB[(i<<2)+3];
		gray = (unsigned char)(0.3*r + 0.59*g + 0.11*b);
		ret[i<<2] = gray;
		ret[(i<<2)+1] = gray;
		ret[(i<<2)+2] = gray;
		ret[(i<<2)+3] = alpha;
	}

	return ret;
}

int main(int argc, char ** args){
	usage(argc);
	cl_uint low_threshold = 20;
	cl_uint high_threshold = 60;
	float sigma = 2.5f;
	if(argc>3){
		sigma = atof(args[3]);
	}
	if(argc>4){
		if(isNumber(args[4])) low_threshold = atoi(args[4]);
	}
	if(argc>5){
		if(isNumber(args[5])) high_threshold = atoi(args[5]);
	}
	int width,height,channels;
	// caricamento immagine in memoria come array di unsigned char
	unsigned char * img = stbi_load(args[1],&width,&height,&channels,STBI_rgb_alpha);
	if (channels != 4){
		fprintf(stderr, "Image with %d channels not supported\n", channels);
		exit(1);
	}
	img = RGBA2grayscale(img,width,height);

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

	cl_kernel drog_k = clCreateKernel(prog, "drog_convolution", &err);
	ocl_check(err, "create kernel convolution");
	cl_kernel suppress_k = clCreateKernel(prog, "non_maxima_suppression", &err);
	ocl_check(err, "create kernel non_maxima_suppression");
	cl_kernel hysteresis_k = clCreateKernel(prog, "hysteresis", &err);
	ocl_check(err, "create kernel hysteresis");
    
	/* get information about the preferred work-group size multiple */
	err = clGetKernelWorkGroupInfo(drog_k, d,
		CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
		sizeof(gws_align_drog), &gws_align_drog, NULL);
	ocl_check(err, "preferred wg multiple for log");

	cl_mem d_kernel_matrix = NULL, d_input = NULL, d_angles = NULL, d_magnitudes = NULL, d_supp_magnitudes = NULL, d_output = NULL;

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

	d_magnitudes = clCreateImage(ctx,
		CL_MEM_READ_WRITE,
		&fmt, &desc,
		NULL,
		&err);
	ocl_check(err, "create image d_magnitudes");

	d_supp_magnitudes = clCreateImage(ctx,
		CL_MEM_READ_WRITE,
		&fmt, &desc,
		NULL,
		&err);
	ocl_check(err, "create image d_supp_magnitudes");

	//DroG matrix x direction
	/*float kernel_matrix[] = {
        0.1129f, 0.7733f, 0.1129f,
        0, 0, 0,
		-0.1129f, -0.7733f, -0.1129f};*/
	float kernel_matrix[] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0};
	cl_int kwidth = 5, kheight = 5;
	computeDrogKernel(kernel_matrix, kwidth, kheight, sigma);
	print_matrix(kernel_matrix, kheight, kwidth);

	d_kernel_matrix = clCreateBuffer(ctx,
	CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	sizeof(kernel_matrix), kernel_matrix,
		&err);
	ocl_check(err, "create buffer d_kernel_matrix");

	d_angles = clCreateBuffer(ctx,
	CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
	sizeof(int)*width*height, NULL,
		&err);
	ocl_check(err, "create buffer d_angles");

	d_output = clCreateBuffer(ctx,
	CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
	dstdata_size, NULL,
		&err);
	ocl_check(err, "create buffer d_output");

	cl_event drog_evt, suppress_evt, hysteresis_evt, map_evt;

	drog_evt = drog_convolution(drog_k, que, d_magnitudes, d_input, d_angles, height, width, d_kernel_matrix, kwidth, kheight);
	suppress_evt = non_maxima_suppression(suppress_k, que, d_magnitudes, d_supp_magnitudes, d_angles, height, width);
	hysteresis_evt = m_hysteresis(hysteresis_k, que, d_output, d_supp_magnitudes, low_threshold, high_threshold, height, width);

	outimg = clEnqueueMapBuffer(que, d_output, CL_TRUE,
		CL_MAP_READ,
		0, dstdata_size,
		1, &hysteresis_evt, &map_evt, &err);
	ocl_check(err, "enqueue map d_output");

	const double runtime_drog_ms = runtime_ms(drog_evt);
	const double runtime_suppress_ms = runtime_ms(suppress_evt);
	const double runtime_hysteresis_ms = runtime_ms(hysteresis_evt);
	const double runtime_map_ms = runtime_ms(map_evt);
	const double total_runtime_ms = runtime_drog_ms+runtime_suppress_ms+runtime_hysteresis_ms+runtime_hysteresis_ms;

	const double drog_bw_gbs = dstdata_size/1.0e6/runtime_drog_ms;
	const double suppress_bw_gbs = dstdata_size/1.0e6/runtime_suppress_ms;
	const double hysteresis_bw_gbs = dstdata_size/1.0e6/runtime_hysteresis_ms;
	const double map_bw_gbs = dstdata_size/1.0e6/runtime_map_ms;

	printf("drog convolution: %dx%d uint in %gms: %g GB/s %g GE/s\n",
		height, width, runtime_drog_ms, drog_bw_gbs, height*width/1.0e6/runtime_drog_ms);
	printf("non maxima suppression: %dx%d magnitudes in %gms: %g GB/s %g GE/s\n",
		height, width, runtime_suppress_ms, suppress_bw_gbs, height*width/1.0e6/runtime_suppress_ms);
	printf("hysteresis: %dx%d uint in %gms: %g GB/s %g GE/s\n",
		height, width, runtime_hysteresis_ms, hysteresis_bw_gbs, height*width/1.0e6/runtime_hysteresis_ms);
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
	clReleaseMemObject(d_angles);
	clReleaseMemObject(d_kernel_matrix);
	clReleaseKernel(drog_k);
	clReleaseKernel(suppress_k);
	clReleaseKernel(hysteresis_k);
	clReleaseProgram(prog);
	clReleaseCommandQueue(que);
	clReleaseContext(ctx);

	stbi_image_free(img);
	return 0;
}
