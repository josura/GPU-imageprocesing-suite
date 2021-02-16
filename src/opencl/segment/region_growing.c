#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <unistd.h>
#include <math.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include"../../../stb/stb_image.h" 
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include"../../../stb/stb_image_write.h" 


#define CL_TARGET_OPENCL_VERSION 120
#define MAX_SEED_TRIES 512
#define MAX_GROWTHS 1024
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
	cl_mem d_region_labels, cl_mem d_explore_labels, cl_mem d_input, cl_uint dist_threshold,
	cl_int curr_region, cl_mem d_unfinished_flag, cl_int nrows, cl_int ncols)
{
	const size_t gws[] = { round_mul_up(ncols, gws_align_region_growing), nrows };
	cl_event region_growing_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(region_growing_k, i++, sizeof(d_region_labels), &d_region_labels);
	ocl_check(err, "set region_growing arg %d", i-1);
	err = clSetKernelArg(region_growing_k, i++, sizeof(d_explore_labels), &d_explore_labels);
	ocl_check(err, "set region_growing arg %d", i-1);
	err = clSetKernelArg(region_growing_k, i++, sizeof(curr_region), &curr_region);
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

cl_event color_regions(cl_kernel color_regions_k, cl_command_queue que,
	cl_mem d_input, cl_mem d_region_labels, cl_mem d_region_colors, cl_mem d_output,
	cl_int nrows, cl_int ncols)
{
	const size_t gws[] = { round_mul_up(ncols, gws_align_region_growing), nrows };
	cl_event color_regions_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(color_regions_k, i++, sizeof(d_region_colors), &d_region_colors);
	ocl_check(err, "set color_regions arg %d", i-1);
	err = clSetKernelArg(color_regions_k, i++, sizeof(d_region_labels), &d_region_labels);
	ocl_check(err, "set color_regions arg %d", i-1);
	err = clSetKernelArg(color_regions_k, i++, sizeof(d_input), &d_input);
	ocl_check(err, "set color_regions arg %d", i-1);
	err = clSetKernelArg(color_regions_k, i++, sizeof(d_output), &d_output);
	ocl_check(err, "set color_regions arg %d", i-1);

	err = clEnqueueNDRangeKernel(que, color_regions_k, 2,
		NULL, gws, NULL,
		0, NULL, &color_regions_evt);

	ocl_check(err, "enqueue color_regions");

	return color_regions_evt;
}

void usage(int argc){
	if(argc<4){
		fprintf(stderr,"Usage: ./region_growing <image.png> num_regions <output.png> [dist_threshold]\n ");
		fprintf(stderr,"the image is an image with 4 channels (R,G,B,transparency)\n");
		fprintf(stderr,"num_regions is the number of regions you want to find in the image\n");
		fprintf(stderr,"dist_threshold is the acceptance level for a pixel to enter a region (default 3)\n");
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
	if(!isNumber(args[2])){
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
    int num_regions = atoi(args[2]);
    printf("Number of regions to find: %d\n", num_regions);
	cl_uint dist_threshold = 3;
	if(argc == 5 && isNumber(args[4])) dist_threshold = atoi(args[4]);
    printf("Distance for acceptance level in merging: %d\n", dist_threshold);

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
	cl_kernel region_growing_k = clCreateKernel(prog, "region_growing", &err);
	ocl_check(err, "create kernel region_growing");
	cl_kernel color_regions_k = clCreateKernel(prog, "color_regions", &err);
	ocl_check(err, "create kernel color_regions");
    
	/* get information about the preferred work-group size multiple */
	err = clGetKernelWorkGroupInfo(region_growing_k, d,
		CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
		sizeof(gws_align_region_growing), &gws_align_region_growing, NULL);
	ocl_check(err, "preferred wg multiple for region_growing");

	cl_mem d_input = NULL, d_output = NULL, d_labimg = NULL, d_region_labels = NULL, 
	d_explore_labels = NULL, d_unfinished_flag = NULL, d_region_colors = NULL;

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

	cl_int * h_region_labels = calloc(width*height, sizeof(cl_int));
	cl_uchar * h_explore_labels = calloc(width*height, sizeof(cl_uchar));

	//for(int i=0; i<20; ++i) printf("Explore label %d: %d\n", i, h_explore_labels[i]);

	d_region_labels = clCreateBuffer(ctx,
	CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
	width*height*sizeof(cl_int), h_region_labels,
		&err);
	ocl_check(err, "create buffer d_region_labels");

	d_explore_labels = clCreateBuffer(ctx,
	CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
	width*height*sizeof(cl_uchar), h_explore_labels,
		&err);
	ocl_check(err, "create buffer d_explore_labels");

	int * host_unfinished_flag = malloc(sizeof(cl_int));
	host_unfinished_flag[0] = 1;

	d_unfinished_flag = clCreateBuffer(ctx,
	CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	sizeof(cl_int), host_unfinished_flag,
		&err);
	ocl_check(err, "create buffer d_explore_labels");

	cl_event RGBtoLAB_evt, region_growing_evt[num_regions+1][MAX_GROWTHS], color_regions_evt, map_img_evt;
    
	RGBtoLAB_evt = RGBtoLAB(RGBtoLAB_k, que, d_labimg, d_input, height, width);

	int * n_iter = calloc(num_regions, sizeof(int));
	srand(time(0));
	cl_uchar4 * h_region_colors = calloc(num_regions, sizeof(cl_uchar4));
	int seed_x, seed_y;
	int seed_found, tries;
	int curr_region;

	cl_uchar4 * img4 = (cl_uchar4*)(img);

	//Start from 1 because 0 is the default region
	for(curr_region=0; curr_region<num_regions; ++curr_region){
		seed_found = 0;
		tries = 0;
		while(!seed_found && tries<MAX_SEED_TRIES){
			seed_x = rand() & (width-1);
			seed_y = rand() & (height-1);
			//printf("Trying seed %d %d with region %d\n", seed_x, seed_y, h_region_labels[seed_y*height+seed_x]);
			if(h_region_labels[seed_y*height+seed_x]==0){
				seed_found = 1;
				h_region_labels[seed_y*height+seed_x] = curr_region+1;
				h_explore_labels[seed_y*height+seed_x] = 1;
				h_region_colors[curr_region] = img4[seed_y*height+seed_x];
				printf("Seed %d is (%d, %d), with color (%d, %d, %d, %d)\n", curr_region+1, seed_x, seed_y, h_region_colors[curr_region].x, h_region_colors[curr_region].y, h_region_colors[curr_region].z, h_region_colors[curr_region].w);
			}
			tries++;
		}
		if(tries >= MAX_SEED_TRIES){
			fprintf(stderr, "Exausted number of tries in finding a seed, exiting region growing with %d regions\n", curr_region+1);
			break;
		}
		clEnqueueWriteBuffer(que, d_region_labels, CL_TRUE, 0, width*height*sizeof(cl_int), h_region_labels, 0, NULL, NULL);
		clEnqueueWriteBuffer(que, d_explore_labels, CL_TRUE, 0, width*height*sizeof(cl_uchar), h_explore_labels, 0, NULL, NULL);
		while(*host_unfinished_flag && n_iter[curr_region]<MAX_GROWTHS){
			*host_unfinished_flag = 0;
			clEnqueueWriteBuffer(que, d_unfinished_flag, CL_TRUE, 0, sizeof(cl_int), host_unfinished_flag, 0, NULL, NULL);
			region_growing_evt[curr_region][n_iter[curr_region]] = region_growing(region_growing_k, que, d_region_labels, 
				d_explore_labels, d_labimg, dist_threshold, curr_region+1, d_unfinished_flag, height, width);
			clEnqueueReadBuffer(que, d_unfinished_flag, CL_TRUE, 0, sizeof(cl_int), host_unfinished_flag, 1, &(region_growing_evt[curr_region][n_iter[curr_region]]), NULL);
			n_iter[curr_region]++;
			//printf("Iterazione %d, flag %d\n", n_iter[curr_region], *host_unfinished_flag);
		}
		err = clEnqueueReadBuffer(que, d_region_labels, CL_TRUE, 0, width*height*sizeof(cl_int), h_region_labels, 0, NULL, NULL);
		err = clEnqueueReadBuffer(que, d_explore_labels, CL_TRUE, 0, width*height*sizeof(cl_uchar), h_explore_labels, 0, NULL, NULL);
		if(n_iter[curr_region]>=MAX_GROWTHS){
			printf("Exausted number of growing cycles (%d) on region %d, going to the next region\n", n_iter[curr_region], curr_region+1);
			continue;
		}	
	}
	int final_num_regions = curr_region;
	printf("Region growing finished, %d regions found!\n", final_num_regions);

	d_region_colors = clCreateBuffer(ctx,
	CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	num_regions*sizeof(cl_uchar4), h_region_colors,
		&err);
	ocl_check(err, "create buffer d_region_colors");
	
	color_regions_evt = color_regions(color_regions_k, que, d_input, d_region_labels, d_region_colors, d_output, height, width);
	
	outimg = clEnqueueMapBuffer(que, d_output, CL_TRUE,
		CL_MAP_READ,
		0, dstdata_size,
		1, &color_regions_evt, &map_img_evt, &err);
	ocl_check(err, "enqueue map d_output");
	
	const double runtime_RGBtoLAB_ms = runtime_ms(RGBtoLAB_evt);
	double runtime_region_growing_ms = 0;
	for(int i=0; i<final_num_regions-1; ++i){
		runtime_region_growing_ms +=  total_runtime_ms(region_growing_evt[i][0], region_growing_evt[i][n_iter[final_num_regions-1]]);
	}
	runtime_region_growing_ms += total_runtime_ms(region_growing_evt[final_num_regions-1][0], region_growing_evt[final_num_regions-1][n_iter[final_num_regions-1]-2]);
	const double runtime_color_regions_ms = runtime_ms(color_regions_evt);
	const double runtime_map_ms = runtime_ms(map_img_evt);

	const double RGBtoLAB_bw_gbs = dstdata_size/1.0e6/runtime_RGBtoLAB_ms;
	const double region_growing_bw_gbs = dstdata_size/1.0e6/runtime_region_growing_ms;
	const double color_regions_bw_gbs = dstdata_size/1.0e6/runtime_color_regions_ms;
	const double map_bw_gbs = dstdata_size/1.0e6/runtime_map_ms;
	
	printf("RGB to CIELAB: %dx%d int in %gms: %g GB/s %g GE/s\n",
		height, width, runtime_RGBtoLAB_ms, RGBtoLAB_bw_gbs, height*width/1.0e6/runtime_RGBtoLAB_ms);
	printf("region growing: %dx%d int in %gms: %g GB/s %g GE/s\n",
		height, width, runtime_region_growing_ms, region_growing_bw_gbs, height*width/1.0e6/runtime_region_growing_ms);
	printf("color regions: %dx%d int in %gms: %g GB/s %g GE/s\n",
		height, width, runtime_color_regions_ms, color_regions_bw_gbs, height*width/1.0e6/runtime_color_regions_ms);
	printf("map img: %dx%d int in %gms: %g GB/s %g GE/s\n",
		dstheight, dstwidth, runtime_map_ms, map_bw_gbs, dstheight*dstwidth/1.0e6/runtime_map_ms);

	char outputImage[128];
	sprintf(outputImage,"%s",args[3]);
	printf("image saved as %s\n",outputImage);
	stbi_write_png(outputImage,dstwidth,dstheight,channels,outimg,channels*dstwidth);

	err = clEnqueueUnmapMemObject(que, d_output, outimg,
		0, NULL, NULL);
	ocl_check(err, "unmap output");

	clReleaseMemObject(d_output);
	clReleaseMemObject(d_input);
	clReleaseMemObject(d_unfinished_flag);
	clReleaseMemObject(d_region_colors);
	clReleaseMemObject(d_region_labels);
	clReleaseMemObject(d_explore_labels);
	clReleaseKernel(region_growing_k);
	clReleaseKernel(color_regions_k);
	clReleaseKernel(RGBtoLAB_k);
	clReleaseProgram(prog);
	clReleaseCommandQueue(que);
	clReleaseContext(ctx);

	stbi_image_free(img);
	free(host_unfinished_flag);
	free(h_region_labels);
	free(h_explore_labels);
	free(n_iter);
	free(h_region_colors);
	return 0;
}
