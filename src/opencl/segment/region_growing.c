#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <ctype.h>

#define STB_IMAGE_IMPLEMENTATION
#include"../../../stb/stb_image.h" 
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include"../../../stb/stb_image_write.h" 


#define CL_TARGET_OPENCL_VERSION 120
#define MAX_SEED_TRIES 512
#define MAX_GROWTHS 4096
#include "../ocl_boiler.h"


size_t gws_align_region_growing;

short isNumber(const char* string){
	int i=0;
	if(string==NULL)return 0;
	while(string[i]!=0){
		if(!isdigit(string[i])){return 0;}
		i++;
	}
	return 1;
}

cl_event RGBtoLAB(cl_kernel RGBtoLAB_k, cl_command_queue que,
	cl_mem d_labimg, cl_mem d_input, cl_int nrows, cl_int ncols)
{
	const size_t gws[] = { round_mul_up(ncols, gws_align_region_growing), nrows };
	cl_event RGBtoLAB_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(RGBtoLAB_k, i++, sizeof(d_input), &d_input);
	ocl_check(err, "set RGBtoLAB arg %d", i-1);
	err = clSetKernelArg(RGBtoLAB_k, i++, sizeof(d_labimg), &d_labimg);
	ocl_check(err, "set RGBtoLAB arg %d", i-1);

	err = clEnqueueNDRangeKernel(que, RGBtoLAB_k, 2,
		NULL, gws, NULL,
		0, NULL, &RGBtoLAB_evt);

	ocl_check(err, "enqueue RGBtoLAB");

	return RGBtoLAB_evt;
}

cl_event region_growing(cl_kernel region_growing_k, cl_command_queue que,
	cl_mem d_explore_labels, cl_mem d_input, float dist_threshold,
	cl_mem d_unfinished_flag, cl_int nrows, cl_int ncols)
{
	const size_t gws[] = { round_mul_up(ncols, gws_align_region_growing), nrows };
	cl_event region_growing_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(region_growing_k, i++, sizeof(d_explore_labels), &d_explore_labels);
	ocl_check(err, "set region_growing arg %d", i-1);
	err = clSetKernelArg(region_growing_k, i++, sizeof(dist_threshold), &dist_threshold);
	ocl_check(err, "set region_growing arg %d", i-1);
	err = clSetKernelArg(region_growing_k, i++, sizeof(d_unfinished_flag), &d_unfinished_flag);
	ocl_check(err, "set region_growing arg %d", i-1);
	err = clSetKernelArg(region_growing_k, i++, sizeof(d_input), &d_input);
	ocl_check(err, "set region_growing arg %d", i-1);

	err = clEnqueueNDRangeKernel(que, region_growing_k, 2,
		NULL, gws, NULL,
		0, NULL, &region_growing_evt);

	ocl_check(err, "enqueue region_growing");

	return region_growing_evt;
}

cl_event color_region(cl_kernel color_region_k, cl_command_queue que,
	cl_mem d_input, cl_mem d_explore_labels, cl_uchar4 h_region_color, cl_mem d_output,
	cl_int nrows, cl_int ncols)
{
	const size_t gws[] = { round_mul_up(ncols, gws_align_region_growing), nrows };
	cl_event color_region_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(color_region_k, i++, sizeof(h_region_color), &h_region_color);
	ocl_check(err, "set color_region arg %d", i-1);
	err = clSetKernelArg(color_region_k, i++, sizeof(d_explore_labels), &d_explore_labels);
	ocl_check(err, "set color_region arg %d", i-1);
	err = clSetKernelArg(color_region_k, i++, sizeof(d_input), &d_input);
	ocl_check(err, "set color_region arg %d", i-1);
	err = clSetKernelArg(color_region_k, i++, sizeof(d_output), &d_output);
	ocl_check(err, "set color_region arg %d", i-1);

	err = clEnqueueNDRangeKernel(que, color_region_k, 2,
		NULL, gws, NULL,
		0, NULL, &color_region_evt);

	ocl_check(err, "enqueue color_region");

	return color_region_evt;
}

void usage(int argc){
	if(argc<5){
		fprintf(stderr,"Usage: ./region_growing <image.png> <output.png> <seed_pixel_x> <seed_pixel_y> [dist_threshold]\n ");
		fprintf(stderr,"the image is an image with 4 channels (R,G,B,transparency)\n");
		fprintf(stderr,"seed_pixel_x and seed_pixel_y are the coordinates of the region seed pixel\n");
		fprintf(stderr,"dist_threshold is the acceptance level for a pixel to enter a region (default 2)\n");
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
	if(!isNumber(args[3]) || !isNumber(args[4])){
		fprintf(stderr, "Parameter not numeric, exiting\n");
		exit(1);
	}
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
    int seed_x = atoi(args[3]);
	int seed_y = atoi(args[4]);
	if(seed_x >= height || seed_y >= width){
		fprintf(stderr, "The seed pixel chosen is out of range, exiting\n");
		exit(1);
	}
	float dist_threshold = 2.0f;
	if(argc > 5) dist_threshold = atof(args[5]);
    printf("Distance for acceptance level in merging: %f\n", dist_threshold);

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
	cl_kernel RGBtoLAB_k = clCreateKernel(prog, "RGBtoLAB", &err);
	ocl_check(err, "create kernel RGBtoLAB");
	cl_kernel region_growing_k = clCreateKernel(prog, "region_growing_single_region", &err);
	ocl_check(err, "create kernel region_growing");
	cl_kernel color_region_k = clCreateKernel(prog, "color_single_region", &err);
	ocl_check(err, "create kernel color_region");
    
	/* get information about the preferred work-group size multiple */
	err = clGetKernelWorkGroupInfo(region_growing_k, d,
		CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
		sizeof(gws_align_region_growing), &gws_align_region_growing, NULL);
	ocl_check(err, "preferred wg multiple for region_growing");

	cl_mem d_input = NULL, d_output = NULL, d_labimg = NULL, d_region_labels = NULL, 
	d_explore_labels = NULL, d_unfinished_flag = NULL;

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

	d_labimg = clCreateImage(ctx,
		CL_MEM_READ_WRITE,
		&fmt, &desc,
		NULL,
		&err);
	ocl_check(err, "create image d_labimg");

	d_output = clCreateBuffer(ctx,
	CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
	dstdata_size, NULL,
		&err);
	ocl_check(err, "create buffer d_output");

	cl_uchar * h_explore_labels = calloc(width*height, sizeof(cl_uchar));

	cl_int * host_unfinished_flag = malloc(sizeof(cl_int));
	host_unfinished_flag[0] = 1;

	d_unfinished_flag = clCreateBuffer(ctx,
	CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
	sizeof(cl_int), host_unfinished_flag,
		&err);
	ocl_check(err, "create buffer d_unfinished_flag");

	cl_event RGBtoLAB_evt, region_growing_evt[MAX_GROWTHS], color_region_evt, map_img_evt;
    
	RGBtoLAB_evt = RGBtoLAB(RGBtoLAB_k, que, d_labimg, d_input, height, width);

	cl_uchar4 * img4 = (cl_uchar4*)(img);
	cl_uchar4 h_region_color = img4[seed_y*height+seed_x];
	h_explore_labels[seed_y*height+seed_x] = 1;

	d_explore_labels = clCreateBuffer(ctx,
	CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
	width*height*sizeof(cl_uchar), h_explore_labels,
		&err);
	ocl_check(err, "create buffer d_explore_labels");
	
	int n_iter = 0;

	while(*host_unfinished_flag && n_iter<MAX_GROWTHS){
		*host_unfinished_flag = 0;
		clEnqueueWriteBuffer(que, d_unfinished_flag, CL_TRUE, 0, sizeof(cl_int), host_unfinished_flag, 0, NULL, NULL);
		region_growing_evt[n_iter] = region_growing(region_growing_k, que, d_explore_labels, d_labimg, dist_threshold, d_unfinished_flag, height, width);
		clEnqueueReadBuffer(que, d_unfinished_flag, CL_TRUE, 0, sizeof(cl_int), host_unfinished_flag, 1, &(region_growing_evt[n_iter]), NULL);
		n_iter++;
	}
	//err = clEnqueueReadBuffer(que, d_explore_labels, CL_TRUE, 0, width*height*sizeof(cl_uchar), h_explore_labels, 0, NULL, NULL);
	if(n_iter>=MAX_GROWTHS){
			printf("Exausted number of growing cycles (%d)\n", n_iter);
	}
	printf("Number of iterations: %d\n", n_iter);
	
	color_region_evt = color_region(color_region_k, que, d_input, d_explore_labels, h_region_color, d_output, height, width);
	
	outimg = clEnqueueMapBuffer(que, d_output, CL_TRUE,
		CL_MAP_READ,
		0, dstdata_size,
		1, &color_region_evt, &map_img_evt, &err);
	ocl_check(err, "enqueue map d_output");
	
	const double runtime_RGBtoLAB_ms = runtime_ms(RGBtoLAB_evt);
	double runtime_region_growing_ms = total_runtime_ms(region_growing_evt[0], region_growing_evt[n_iter-1]);
	const double runtime_color_region_ms = runtime_ms(color_region_evt);
	const double runtime_map_ms = runtime_ms(map_img_evt);
	const double total_runtime_ms = runtime_RGBtoLAB_ms+runtime_region_growing_ms+runtime_color_region_ms+runtime_map_ms;

	const double RGBtoLAB_bw_gbs = dstdata_size/1.0e6/runtime_RGBtoLAB_ms;
	const double region_growing_bw_gbs = dstdata_size/1.0e6/runtime_region_growing_ms;
	const double color_region_bw_gbs = dstdata_size/1.0e6/runtime_color_region_ms;
	const double map_bw_gbs = dstdata_size/1.0e6/runtime_map_ms;
	
	printf("RGB to CIELAB: %dx%d int in %gms: %g GB/s %g GE/s\n",
		height, width, runtime_RGBtoLAB_ms, RGBtoLAB_bw_gbs, height*width/1.0e6/runtime_RGBtoLAB_ms);
	printf("region growing: %dx%d int in %gms: %g GB/s %g GE/s\n",
		height, width, runtime_region_growing_ms, region_growing_bw_gbs, height*width/1.0e6/runtime_region_growing_ms);
	printf("color regions: %dx%d int in %gms: %g GB/s %g GE/s\n",
		height, width, runtime_color_region_ms, color_region_bw_gbs, height*width/1.0e6/runtime_color_region_ms);
	printf("map img: %dx%d int in %gms: %g GB/s %g GE/s\n",
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
	clReleaseMemObject(d_unfinished_flag);
	clReleaseMemObject(d_explore_labels);
	clReleaseKernel(region_growing_k);
	clReleaseKernel(color_region_k);
	clReleaseKernel(RGBtoLAB_k);
	clReleaseProgram(prog);
	clReleaseCommandQueue(que);
	clReleaseContext(ctx);

	stbi_image_free(img);
	free(host_unfinished_flag);
	return 0;
}
