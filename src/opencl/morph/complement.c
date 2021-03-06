#include<stdio.h>
#include<string.h>

#define STB_IMAGE_IMPLEMENTATION
#include"../../../stb/stb_image.h" 
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include"../../../stb/stb_image_write.h" 


#define CL_TARGET_OPENCL_VERSION 120
#include "../ocl_boiler.h"


size_t gws_align_complement;

cl_event complement(cl_kernel complement_k, cl_command_queue que,
	cl_mem d_output, cl_mem d_input,
	cl_int nrows, cl_int ncols)
{
	const size_t gws[] = { round_mul_up(ncols, gws_align_complement), nrows };
	cl_event complement_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(complement_k, i++, sizeof(d_output), &d_output);
	ocl_check(err, "set complement arg", i-1);
	err = clSetKernelArg(complement_k, i++, sizeof(d_input), &d_input);
	ocl_check(err, "set complement arg", i-1);

	err = clEnqueueNDRangeKernel(que, complement_k, 2,
		NULL, gws, NULL,
		0, NULL, &complement_evt);

	ocl_check(err, "enqueue complement");

	return complement_evt;
}



void usage(int argc){
	if(argc<3){
		fprintf(stderr,"Usage: ./complement <image.png> <output.png> \n");
		fprintf(stderr,"the image is an image with 4 channels (R,G,B,transparency)\n");
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
	int width,height,channels;
	// caricamento immagine in memoria come array di unsigned char
	unsigned char * img= stbi_load(args[1],&width,&height,&channels,STBI_rgb_alpha);
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
	cl_program prog = create_program("morphology.ocl", ctx, d);
	int err=0;
	cl_kernel complement_k = NULL;
	complement_k = clCreateKernel(prog, "complement", &err);
	ocl_check(err, "create kernel complement image");
	/* get information about the preferred work-group size multiple */
	err = clGetKernelWorkGroupInfo(complement_k, d,
		CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
		sizeof(gws_align_complement), &gws_align_complement, NULL);
	ocl_check(err, "preferred wg multiple for complement");

	cl_mem d_input = NULL, d_output = NULL;

	const cl_image_format fmt = {
		.image_channel_order = CL_RGBA,
		.image_channel_data_type = CL_UNORM_INT8,
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



	d_output = clCreateBuffer(ctx,
		CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
		dstdata_size, NULL,
		&err);
	ocl_check(err, "create buffer d_output");

	cl_event complement_evt, map_evt;

	complement_evt = complement(complement_k, que, d_output, d_input, height, width);

	outimg = clEnqueueMapBuffer(que, d_output, CL_FALSE,
		CL_MAP_READ,
		0, dstdata_size,
		1, &complement_evt, &map_evt, &err);
	ocl_check(err, "enqueue map d_output");

	err = clWaitForEvents(1, &map_evt);
	ocl_check(err, "clfinish");

	const double runtime_complement_ms = runtime_ms(complement_evt);
	const double runtime_map_ms = runtime_ms(map_evt);

	const double complement_bw_gbs = dstdata_size/1.0e6/runtime_complement_ms;
	const double map_bw_gbs = dstdata_size/1.0e6/runtime_map_ms;

	printf("complement: %dx%d int in %gms: %g GB/s %g GE/s\n",
		height, width, runtime_complement_ms, complement_bw_gbs, height*width/1.0e6/runtime_complement_ms);
	printf("map: %dx%d int in %gms: %g GB/s %g GE/s\n",
		dstheight, dstwidth, runtime_map_ms, map_bw_gbs, dstheight*dstwidth/1.0e6/runtime_map_ms);

	char outputImage[128];
	sprintf(outputImage,"%s",args[2]);
	printf("%s\n",outputImage);
	stbi_write_png(outputImage,dstwidth,dstheight,channels,outimg,channels*dstwidth);
	err = clEnqueueUnmapMemObject(que, d_output, outimg,
		0, NULL, NULL);
	ocl_check(err, "unmap output");

	clReleaseMemObject(d_output);
	clReleaseMemObject(d_input);
	clReleaseKernel(complement_k);
	clReleaseProgram(prog);
	clReleaseCommandQueue(que);
	clReleaseContext(ctx);








	stbi_image_free(img);
	
	return 0;
}
