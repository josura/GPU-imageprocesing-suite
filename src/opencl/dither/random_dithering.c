#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <unistd.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include"../../../stb/stb_image.h" 
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include"../../../stb/stb_image_write.h" 


#define CL_TARGET_OPENCL_VERSION 120
#include "../ocl_boiler.h"


size_t gws_align_dithering;

//Get one of the seeds
static inline uint64_t rdtsc(void)
{
	uint64_t val;
	uint32_t h, l;
    __asm__ __volatile__("rdtsc" : "=a" (l), "=d" (h));
        val = ((uint64_t)l) | (((uint64_t)h) << 32);
        return val;
}

short isNumber(const char* string){
	int i=0;
	if(string==NULL)return 0;
	while(string[i]!=0){
		if(!isdigit(string[i])){return 0;}
		i++;
	}
	return 1;
}

cl_event dithering(cl_kernel dithering_k, cl_command_queue que,
	cl_mem d_output, cl_mem d_input, cl_uint4 seeds,
	cl_int nrows, cl_int ncols, cl_uchar num_levels)
{
	const size_t gws[] = { round_mul_up(ncols, gws_align_dithering), nrows };
	cl_event dithering_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(dithering_k, i++, sizeof(d_output), &d_output);
	ocl_check(err, "set dithering arg %d", i-1);
	err = clSetKernelArg(dithering_k, i++, sizeof(d_input), &d_input);
	ocl_check(err, "set dithering arg %d", i-1);
	err = clSetKernelArg(dithering_k, i++, sizeof(num_levels), &num_levels);
	ocl_check(err, "set dithering arg %d", i-1);
	err = clSetKernelArg(dithering_k, i++, sizeof(seeds), &seeds);
	ocl_check(err, "set dithering arg %d", i-1);

	err = clEnqueueNDRangeKernel(que, dithering_k, 2,
		NULL, gws, NULL,
		0, NULL, &dithering_evt);

	ocl_check(err, "enqueue dithering");

	return dithering_evt;
}



void usage(int argc){
	if(argc<4){
		fprintf(stderr,"Usage: ./random_dithering <image.png> num_levels <output.png>\n ");
		fprintf(stderr,"the image is an image with 4 channels (R,G,B,transparency)\n");
		fprintf(stderr,"num_levels is the new number of levels (max 256)\n");
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
    cl_uchar num_levels = atoi(args[2]);
    printf("New number of levels: %d\n", num_levels);

	unsigned char * outimg = NULL;
	int data_size=width*height*channels;
	int dstwidth=width,dstheight=height;
	int dstdata_size=dstwidth*dstheight*channels;
	cl_platform_id p = select_platform();
	cl_device_id d = select_device(p);
	cl_context ctx = create_context(p, d);
	cl_command_queue que = create_queue(ctx, d);
	cl_program prog = create_program("dithering.ocl", ctx, d);
	int err=0;
	cl_kernel dithering_k = clCreateKernel(prog, "random_dithering", &err);
	ocl_check(err, "create kernel random_dithering");
    
	/* get information about the preferred work-group size multiple */
	err = clGetKernelWorkGroupInfo(dithering_k, d,
		CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
		sizeof(gws_align_dithering), &gws_align_dithering, NULL);
	ocl_check(err, "preferred wg multiple for dithering");

	cl_mem d_input = NULL, d_output = NULL;

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

	//seeds for the edited MWC64X
	cl_uint4 seeds = {.x = time(0) & 134217727, .y = (getpid() * getpid() * getpid()) & 134217727, .z = (clock()*clock()) & 134217727, .w = rdtsc() & 134217727};

	printf("Seeds: %d, %d, %d, %d\n", seeds.x, seeds.y, seeds.z, seeds.w);

	d_output = clCreateBuffer(ctx,
	CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
	dstdata_size, NULL,
		&err);
	ocl_check(err, "create buffer d_output");

	cl_event dithering_evt, map_evt;
    
	dithering_evt = dithering(dithering_k, que, d_output, d_input, seeds, height, width, num_levels);
    //clWaitForEvents(1, &dithering_evt);
    //printf("Fin\n");
    //return;

	outimg = clEnqueueMapBuffer(que, d_output, CL_TRUE,
		CL_MAP_READ,
		0, dstdata_size,
		1, &dithering_evt, &map_evt, &err);
	ocl_check(err, "enqueue map d_output");

	const double runtime_dithering_ms = runtime_ms(dithering_evt);
	const double runtime_map_ms = runtime_ms(map_evt);

	const double dithering_bw_gbs = dstdata_size/1.0e6/runtime_dithering_ms;
	const double map_bw_gbs = dstdata_size/1.0e6/runtime_map_ms;

	printf("random dithering: %dx%d int in %gms: %g GB/s %g GE/s\n",
		height, width, runtime_dithering_ms, dithering_bw_gbs, height*width/1.0e6/runtime_dithering_ms);
	printf("map: %dx%d int in %gms: %g GB/s %g GE/s\n",
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
	clReleaseKernel(dithering_k);
	clReleaseProgram(prog);
	clReleaseCommandQueue(que);
	clReleaseContext(ctx);

	stbi_image_free(img);
	return 0;
}
