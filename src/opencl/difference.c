#include<stdio.h>
#include<string.h>

#define STB_IMAGE_IMPLEMENTATION
#include"../../stb/stb_image.h" 
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include"../../stb/stb_image_write.h" 


#define CL_TARGET_OPENCL_VERSION 120
#include "ocl_boiler.h"


size_t gws_align_difference;

cl_event difference(cl_kernel difference_k, cl_command_queue que,
	cl_mem d_output, cl_mem d_input,cl_mem d_input2,
	cl_int nrows, cl_int ncols,cl_int input2_rows,cl_int input2_cols)
{
	const size_t gws[] = { round_mul_up(ncols, gws_align_difference), nrows };
	cl_event difference_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(difference_k, i++, sizeof(d_output), &d_output);
	ocl_check(err, "set difference arg", i-1);
	err = clSetKernelArg(difference_k, i++, sizeof(d_input), &d_input);
	ocl_check(err, "set difference arg", i-1);
	err = clSetKernelArg(difference_k, i++, sizeof(d_input2), &d_input2);
	ocl_check(err, "set difference arg", i-1);

	err = clEnqueueNDRangeKernel(que, difference_k, 2,
		NULL, gws, NULL,
		0, NULL, &difference_evt);

	ocl_check(err, "enqueue difference");

	return difference_evt;
}



void usage(int argc){
	if(argc<4){
		fprintf(stderr,"Usage: ./difference <image1.png> <image2.png>\n");
		fprintf(stderr,"the image1 is an image with 4 channels (R,G,B,transparency)\n");
		fprintf(stderr,"image2 is an image that is substracted to image1\n");
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
	cl_kernel difference_k = NULL;
	difference_k = clCreateKernel(prog, "imageDifference", &err);
	ocl_check(err, "create kernel difference image");
	/* get information about the preferred work-group size multiple */
	err = clGetKernelWorkGroupInfo(difference_k, d,
		CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
		sizeof(gws_align_difference), &gws_align_difference, NULL);
	ocl_check(err, "preferred wg multiple for difference");

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

	cl_mem d_input2=NULL;

	int input2_width=3,input2_height=3,input2_channels=4;
	unsigned char * imginput2= stbi_load(args[2],&input2_width,&input2_height,&input2_channels,STBI_rgb_alpha);
	if(imginput2==NULL){
		printf("error loading of input2 image\n");
		exit(1);
	}
	printf("image input2 loaded with width %i, height %i and channels %i\n",input2_width,input2_height,input2_channels);
	if (input2_channels < 3) {
                fprintf(stderr, "source input2 must have 4 channels\n");
                exit(1);
        }
	if (input2_width!=width || input2_height!=height){
		fprintf(stderr,"the images are not of the same size\n");
		exit(1);
	}


	const cl_image_format fmt_input2 = {
		.image_channel_order = CL_RGBA,
		.image_channel_data_type = CL_UNORM_INT8,
	};
	const cl_image_desc input2_desc = {
		.image_type = CL_MEM_OBJECT_IMAGE2D,
		.image_width = input2_width,
		.image_height = input2_height,
		//.image_row_pitch = src.data_size/src.height,
	};
	d_input2 = clCreateImage(ctx,
		CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
		&fmt_input2, &input2_desc,
		imginput2,
		&err);
	ocl_check(err, "create image d_input");

	d_output = clCreateBuffer(ctx,
		CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
		dstdata_size, NULL,
		&err);
	ocl_check(err, "create buffer d_output");

	cl_event difference_evt, map_evt;

	difference_evt = difference(difference_k, que, d_output, d_input,d_input2, height, width, input2_height,input2_width);

	outimg = clEnqueueMapBuffer(que, d_output, CL_FALSE,
		CL_MAP_READ,
		0, dstdata_size,
		1, &difference_evt, &map_evt, &err);
	ocl_check(err, "enqueue map d_output");

	err = clWaitForEvents(1, &map_evt);
	ocl_check(err, "clfinish");

	const double runtime_difference_ms = runtime_ms(difference_evt);
	const double runtime_map_ms = runtime_ms(map_evt);

	const double difference_bw_gbs = dstdata_size/1.0e6/runtime_difference_ms;
	const double map_bw_gbs = dstdata_size/1.0e6/runtime_map_ms;

	printf("difference: %dx%d int in %gms: %g GB/s %g GE/s\n",
		height, width, runtime_difference_ms, difference_bw_gbs, height*width/1.0e6/runtime_difference_ms);
	printf("map: %dx%d int in %gms: %g GB/s %g GE/s\n",
		dstheight, dstwidth, runtime_map_ms, map_bw_gbs, dstheight*dstwidth/1.0e6/runtime_map_ms);

	char outputImage[128];
	sprintf(outputImage,"%s",args[3]);
	printf("%s\n",outputImage);
	stbi_write_png(outputImage,dstwidth,dstheight,channels,outimg,channels*dstwidth);
	

	err = clEnqueueUnmapMemObject(que, d_output, outimg,
		0, NULL, NULL);
	ocl_check(err, "unmap output");

	clReleaseMemObject(d_output);
	clReleaseMemObject(d_input);
	clReleaseKernel(difference_k);
	clReleaseProgram(prog);
	clReleaseCommandQueue(que);
	clReleaseContext(ctx);








	stbi_image_free(img);
	
	return 0;
}
