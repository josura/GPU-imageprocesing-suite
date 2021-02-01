#include<stdio.h>
#include<string.h>

#define STB_IMAGE_IMPLEMENTATION
#include"../../stb/stb_image.h" 
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include"../../stb/stb_image_write.h" 


#define CL_TARGET_OPENCL_VERSION 120
#include "ocl_boiler.h"


size_t gws_align_dilation;

cl_event dilation(cl_kernel dilation_k, cl_command_queue que,
	cl_mem d_output, cl_mem d_input,cl_mem d_strel,
	cl_int nrows, cl_int ncols,cl_int strel_rows,cl_int strel_cols)
{
	const size_t gws[] = { round_mul_up(ncols, gws_align_dilation), nrows };
	cl_event dilation_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(dilation_k, i++, sizeof(d_output), &d_output);
	ocl_check(err, "set dilation arg", i-1);
	err = clSetKernelArg(dilation_k, i++, sizeof(d_input), &d_input);
	ocl_check(err, "set dilation arg", i-1);
	err = clSetKernelArg(dilation_k, i++, sizeof(d_strel), &d_strel);
	ocl_check(err, "set dilation arg", i-1);

	err = clEnqueueNDRangeKernel(que, dilation_k, 2,
		NULL, gws, NULL,
		0, NULL, &dilation_evt);

	ocl_check(err, "enqueue dilation");

	return dilation_evt;
}



void usage(int argc){
	if(argc<3){
		fprintf(stderr,"Usage: ./dilation <image.png> <strel.png> [b/g]");
		fprintf(stderr,"the image is an image with 4 channels (R,G,B,transparency)");
		fprintf(stderr,"strel is an image that is used as a structuring element");
		fprintf(stderr,"b is for binary dilation, while g is for grayscale dilation");

		exit(1);
	}
}

int main(int argc, char ** args){
	usage(argc);
	if(args[3][0]!='b' && args[3][0]!='g'){
		
		fprintf(stderr,"%s option not supported\n",args[3]);
		exit(1);

	}
	int width,height,channels;
	// caricamento immagine in memoria come array di unsigned char
	unsigned char * img= stbi_load(args[1],&width,&height,&channels,STBI_rgb_alpha);
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
	cl_kernel dilation_k = NULL;
	switch(args[3][0]){
	case 'g':
		dilation_k = clCreateKernel(prog, "dilationImage", &err);
		ocl_check(err, "create kernel dilation image");
		break;
	case 'b':
		dilation_k = clCreateKernel(prog, "dilationImageBinary", &err);
		ocl_check(err, "create kernel dilation binary image");
		break;
	}
	/* get information about the preferred work-group size multiple */
	err = clGetKernelWorkGroupInfo(dilation_k, d,
		CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
		sizeof(gws_align_dilation), &gws_align_dilation, NULL);
	ocl_check(err, "preferred wg multiple for dilation");

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


	//SEZIONE STRUCTURING ELEMENT
	cl_mem d_strel=NULL;
	

	/**STRUCTURING ELEMENT DEFINITO A MANO **/
	int strel_width=3,strel_height=3,strel_channels=4;
	//unsigned char * strelptr = malloc(strel_width*strel_height*channels);
	//strelptr[0]
	

	/** STRUCTURING ELEMENT DA IMMAGINE **/
	unsigned char * imgstrel= stbi_load(args[2],&strel_width,&strel_height,&strel_channels,STBI_rgb_alpha);
	if(imgstrel==NULL){
		printf("error loading of strel image\n");
		exit(1);
	}
	printf("image strel loaded with width %i, height %i and channels %i\n",strel_width,strel_height,strel_channels);
	if (strel_channels < 3) {
                fprintf(stderr, "source strel must have 4 channels\n");
                exit(1);
        }


	const cl_image_format fmt_strel = {
		.image_channel_order = CL_RGBA,
		.image_channel_data_type = CL_UNORM_INT8,
	};
	const cl_image_desc strel_desc = {
		.image_type = CL_MEM_OBJECT_IMAGE2D,
		.image_width = strel_width,
		.image_height = strel_height,
		//.image_row_pitch = src.data_size/src.height,
	};
	d_strel = clCreateImage(ctx,
		CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
		&fmt_strel, &strel_desc,
		imgstrel,
		&err);
	ocl_check(err, "create image d_input");

	//FINE SEZIONE STRUCTURING ELEMENT
	d_output = clCreateBuffer(ctx,
		CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
		dstdata_size, NULL,
		&err);
	ocl_check(err, "create buffer d_output");

	cl_event dilation_evt, map_evt;

	dilation_evt = dilation(dilation_k, que, d_output, d_input,d_strel, height, width, strel_height,strel_width);

	outimg = clEnqueueMapBuffer(que, d_output, CL_FALSE,
		CL_MAP_READ,
		0, dstdata_size,
		1, &dilation_evt, &map_evt, &err);
	ocl_check(err, "enqueue map d_output");

	err = clWaitForEvents(1, &map_evt);
	ocl_check(err, "clfinish");

	const double runtime_dilation_ms = runtime_ms(dilation_evt);
	const double runtime_map_ms = runtime_ms(map_evt);

	const double dilation_bw_gbs = dstdata_size/1.0e6/runtime_dilation_ms;
	const double map_bw_gbs = dstdata_size/1.0e6/runtime_map_ms;

	printf("dilation: %dx%d int in %gms: %g GB/s %g GE/s\n",
		height, width, runtime_dilation_ms, dilation_bw_gbs, height*width/1.0e6/runtime_dilation_ms);
	printf("map: %dx%d int in %gms: %g GB/s %g GE/s\n",
		dstheight, dstwidth, runtime_map_ms, map_bw_gbs, dstheight*dstwidth/1.0e6/runtime_map_ms);

	char outputImage[128];
	sprintf(outputImage,"processed.png");
	printf("%s\n",outputImage);
	stbi_write_png(outputImage,dstwidth,dstheight,channels,outimg,channels*dstwidth);
	//err = save_pam(output_fname, &dst);
	//if (err != 0) {
	//	fprintf(stderr, "error writing %s\n", output_fname);
	//	exit(1);
	//}

	err = clEnqueueUnmapMemObject(que, d_output, outimg,
		0, NULL, NULL);
	ocl_check(err, "unmap output");

	clReleaseMemObject(d_output);
	clReleaseMemObject(d_input);
	clReleaseKernel(dilation_k);
	clReleaseProgram(prog);
	clReleaseCommandQueue(que);
	clReleaseContext(ctx);








	stbi_image_free(img);
	
	return 0;
}
