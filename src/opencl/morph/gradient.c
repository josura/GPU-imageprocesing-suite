#include"morphology.h"

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
	cl_kernel erosion_k = NULL,dilation_k=NULL,difference_k=NULL;
	erosion_k = clCreateKernel(prog, "erosionImage", &err);
	ocl_check(err, "create kernel erosion image");
	dilation_k = clCreateKernel(prog, "dilationImage", &err);
    ocl_check(err, "create kernel dilation image");
	difference_k = clCreateKernel(prog, "imageDifference", &err);
    ocl_check(err, "create kernel difference image");
	/* get information about the preferred work-group size multiple */
	err = clGetKernelWorkGroupInfo(erosion_k, d,
		CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
		sizeof(gws_align), &gws_align, NULL);
	ocl_check(err, "preferred wg multiple for erosion");

	cl_mem d_input = NULL, d_output = NULL, d_tmp = NULL;

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

	cl_event erosion_evt,dilation_evt,difference_evt, map_evt;

	// DILATION
	dilation_evt = dilation(dilation_k, que, d_output, d_input,d_strel, height, width, strel_height,strel_width);

	outimg = clEnqueueMapBuffer(que, d_output, CL_FALSE,
		CL_MAP_READ,
		0, dstdata_size,
		1, &dilation_evt, &map_evt, &err);
	ocl_check(err, "enqueue map d_output");

	err = clWaitForEvents(1, &map_evt);
	ocl_check(err, "clfinish");

    d_tmp = clCreateImage(ctx,
		CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
		&fmt, &desc,
		outimg,
		&err);
	ocl_check(err, "create image d_tmp");

	
	// EROSION

	erosion_evt = erosion(erosion_k, que, d_output, d_input,d_strel, height, width, strel_height,strel_width);

	outimg = clEnqueueMapBuffer(que, d_output, CL_FALSE,
		CL_MAP_READ,
		0, dstdata_size,
		1, &erosion_evt, &map_evt, &err);
	ocl_check(err, "enqueue map d_output");

	err = clWaitForEvents(1, &map_evt);
	ocl_check(err, "clfinish");

    clReleaseMemObject(d_input);
 
	d_input = clCreateImage(ctx,
		CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
		&fmt, &desc,
		outimg,
		&err);
	ocl_check(err, "create image d_input after erosion");


	difference_evt = difference(difference_k, que, d_output, d_tmp,d_input, height, width, height,width);

    outimg = clEnqueueMapBuffer(que, d_output, CL_FALSE,
		CL_MAP_READ,
		0, dstdata_size,
		1, &erosion_evt, &map_evt, &err);
	ocl_check(err, "enqueue map d_output");

	err = clWaitForEvents(1, &map_evt);
	ocl_check(err, "clfinish");

	const double runtime_erosion_ms = runtime_ms(erosion_evt);
	const double runtime_dilation_ms = runtime_ms(dilation_evt);
	const double runtime_difference_ms = runtime_ms(difference_evt);
	const double runtime_map_ms = runtime_ms(map_evt);

	const double erosion_bw_gbs = dstdata_size/1.0e6/runtime_erosion_ms;
	const double dilation_bw_gbs = dstdata_size/1.0e6/runtime_dilation_ms;
	const double difference_bw_gbs = dstdata_size/1.0e6/runtime_difference_ms;
	const double map_bw_gbs = dstdata_size/1.0e6/runtime_map_ms;

	printf("erosion: %dx%d int in %gms: %g GB/s %g GE/s\n",
		height, width, runtime_erosion_ms, erosion_bw_gbs, height*width/1.0e6/runtime_erosion_ms);
	printf("dilation: %dx%d int in %gms: %g GB/s %g GE/s\n",
		height, width, runtime_dilation_ms, dilation_bw_gbs, height*width/1.0e6/runtime_dilation_ms);
	printf("difference: %dx%d int in %gms: %g GB/s %g GE/s\n",
		height, width, runtime_difference_ms, difference_bw_gbs, height*width/1.0e6/runtime_difference_ms);
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
	clReleaseKernel(erosion_k);
	clReleaseKernel(dilation_k);
	clReleaseKernel(difference_k);
	clReleaseProgram(prog);
	clReleaseCommandQueue(que);
	clReleaseContext(ctx);








	stbi_image_free(img);
	
	return 0;
}
