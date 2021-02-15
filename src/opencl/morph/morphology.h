#include<stdio.h>
#include<string.h>

#define STB_IMAGE_IMPLEMENTATION
#include"../../../stb/stb_image.h" 
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include"../../../stb/stb_image_write.h" 


#define CL_TARGET_OPENCL_VERSION 120
#include "../ocl_boiler.h"


size_t gws_align;

cl_event erosion(cl_kernel erosion_k, cl_command_queue que,
	cl_mem d_output, cl_mem d_input,cl_mem d_strel,
	cl_int nrows, cl_int ncols,cl_int strel_rows,cl_int strel_cols)
{
	const size_t gws[] = { round_mul_up(ncols, gws_align), nrows };
	cl_event erosion_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(erosion_k, i++, sizeof(d_output), &d_output);
	ocl_check(err, "set erosion arg", i-1);
	err = clSetKernelArg(erosion_k, i++, sizeof(d_input), &d_input);
	ocl_check(err, "set erosion arg", i-1);
	err = clSetKernelArg(erosion_k, i++, sizeof(d_strel), &d_strel);
	ocl_check(err, "set erosion arg", i-1);

	err = clEnqueueNDRangeKernel(que, erosion_k, 2,
		NULL, gws, NULL,
		0, NULL, &erosion_evt);

	ocl_check(err, "enqueue erosion");

	return erosion_evt;
}

cl_event dilation(cl_kernel dilation_k, cl_command_queue que,
	cl_mem d_output, cl_mem d_input,cl_mem d_strel,
	cl_int nrows, cl_int ncols,cl_int strel_rows,cl_int strel_cols)
{
	const size_t gws[] = { round_mul_up(ncols, gws_align), nrows };
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

cl_event complement(cl_kernel complement_k, cl_command_queue que,
	cl_mem d_output, cl_mem d_input,
	cl_int nrows, cl_int ncols)
{
	const size_t gws[] = { round_mul_up(ncols, gws_align), nrows };
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

cl_event difference(cl_kernel difference_k, cl_command_queue que,
	cl_mem d_output, cl_mem d_input,cl_mem d_input2,
	cl_int nrows, cl_int ncols,cl_int input2_rows,cl_int input2_cols)
{
	const size_t gws[] = { round_mul_up(ncols, gws_align), nrows };
	cl_event difference_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(difference_k, i++, sizeof(d_output), &d_output);
	ocl_check(err, "set difference arg d_output", i-1);
	err = clSetKernelArg(difference_k, i++, sizeof(d_input), &d_input);
	ocl_check(err, "set difference arg d_input", i-1);
	err = clSetKernelArg(difference_k, i++, sizeof(d_input2), &d_input2);
	ocl_check(err, "set difference arg d_input2", i-1);

	err = clEnqueueNDRangeKernel(que, difference_k, 2,
		NULL, gws, NULL,
		0, NULL, &difference_evt);

	ocl_check(err, "enqueue difference");

	return difference_evt;
}



unsigned char* arrayOfMaxValuesUC(unsigned int dim){
	unsigned char* ret=(unsigned char*)malloc(sizeof(float)*dim);
	for(int i=0;i<dim;i++)
		ret[i]=0xff;
	return ret;
}

unsigned char* grayscale2RGBA(unsigned char* inputGray,int width, int height){
	unsigned int dimension = width*height*sizeof(float);
	unsigned char* ret=(unsigned char*)malloc(dimension*4);
	unsigned char* maxValues = arrayOfMaxValuesUC(dimension);
	memcpy(ret,inputGray,dimension);
	memcpy(ret+dimension,inputGray,dimension);
	memcpy(ret+2*dimension,inputGray,dimension);
	memcpy(ret+3*dimension,maxValues,dimension);

	free(maxValues);
	free(inputGray);
	return ret;
}

unsigned char* GA2RGBA(unsigned char* inputGrayAlpha,int width, int height){
	unsigned int dimension = width*height;
	unsigned char* ret=(unsigned char*)malloc(sizeof(float)*dimension*4);
	unsigned char* maxValues = arrayOfMaxValuesUC(dimension);

	memcpy(ret,inputGrayAlpha,dimension*sizeof(float));
	memcpy(ret+dimension*sizeof(float),inputGrayAlpha,dimension*sizeof(float));
	memcpy(ret+2*dimension*sizeof(float),inputGrayAlpha,dimension*sizeof(float));
	memcpy(ret+3*dimension*sizeof(float),inputGrayAlpha,dimension*sizeof(float));

	free(maxValues);
	free(inputGrayAlpha);
	return ret;
}

unsigned char* RGB2RGBA(unsigned char* inputRGB,int width, int height){
	unsigned int dimension = width*height;
	unsigned char* ret=(unsigned char*)malloc(sizeof(float)*dimension*4);
	unsigned char* maxValues = arrayOfMaxValuesUC(dimension);

	memcpy(ret,inputRGB,dimension*sizeof(float));
	memcpy(ret+dimension*sizeof(float),inputRGB,dimension*sizeof(float));
	memcpy(ret+2*dimension*sizeof(float),inputRGB,dimension*sizeof(float));
	memcpy(ret+3*dimension*sizeof(float),maxValues,dimension*sizeof(float));

	free(maxValues);
	free(inputRGB);
	return ret;
}


//TODO not yet supported, only grayscale works somehow
unsigned char* controlChannels(int imagechannels,int strelchannels, unsigned char* image, unsigned char* strel,int imagewidth,int imageheight,int strelwidth,int strelheight){
	if(imagechannels==1){
		printf("grayscale image with only one channel, doing transformation to 4 channels");
		return grayscale2RGBA(image,imagewidth,imageheight);
	}
	if(strelchannels==1){
		printf("grayscale strel image with only one channel, doing transformation to 4 channels");
		return grayscale2RGBA(strel,strelwidth,strelheight);
		
	}
	if(imagechannels==2){
		printf("grayscale image with only one channel, doing transformation to 4 channels");
		return GA2RGBA(image,imagewidth,imageheight);
	}
	if(strelchannels==2){
		printf("grayscale strel image with only one channel, doing transformation to 4 channels");
		return GA2RGBA(strel,strelwidth,strelheight);
		
	}

	if(imagechannels==3){
		printf("grayscale image with only one channel, doing transformation to 4 channels");
		return RGB2RGBA(image,imagewidth,imageheight);
	}
	if(strelchannels==3){
		printf("grayscale strel image with only one channel, doing transformation to 4 channels");
		return RGB2RGBA(strel,strelwidth,strelheight);
		
	}

	/*if (imagechannels < 3) {
                fprintf(stderr, "source image must have 4 channels (<RGB,alpha> or some other format with transparency and 3 channels for color space)\n");
                exit(1);
        }*/
}

void loggingChannels(int imagechannels,int strelchannels){
	if(imagechannels==1){
		printf("grayscale image with only one channel\n");
	}
	if(strelchannels==1){
		printf("grayscale strel image with only one channel\n");
		
	}
	if(imagechannels==2){
		printf("grayscale alpha image with only one channel and transparency\n");
	}
	if(strelchannels==2){
		printf("grayscale alpha strel image with only one channel and transparency\n");
		
	}

	if(imagechannels==3){
		printf("RGB image with 3 channels\n");
	}
	if(strelchannels==3){
		printf("RGB strel image with 3 channels\n");
		
	}
}

unsigned char * fullErosion(unsigned char* image,unsigned char* strel,int imagewidth,int imageheight,int imagechannels,int strelwidth, int strelheight,int strelchannels){

	loggingChannels(imagechannels,strelchannels);

	if(image==NULL){
		printf("error while loading the image, probably image does not exists\n");
		exit(1);
	}
	if(strel==NULL){
		printf("error while loading the image strel, probably image does not exists\n");
		exit(1);
	}
	printf("image loaded with  %i width, %i height and %i channels\n",imagewidth,imageheight,imagechannels);
	if (imagechannels <= 3) {
                fprintf(stderr, "source image must have 4 channels (<RGB,alpha> or some other format with transparency and 3 channels for color space)\n");
                exit(1);
        }
	unsigned char * outimg = NULL;
	int data_size=imagewidth*imageheight*imagechannels;
	int dstwidth=imagewidth,dstheight=imageheight;
	int dstdata_size=dstwidth*dstheight*imagechannels;
	cl_platform_id p = select_platform();
	cl_device_id d = select_device(p);
	cl_context ctx = create_context(p, d);
	cl_command_queue que = create_queue(ctx, d);
	cl_program prog = create_program("morphology.ocl", ctx, d);
	int err=0;
	cl_kernel erosion_k = NULL;
	erosion_k = clCreateKernel(prog, "erosionImage", &err);
	ocl_check(err, "create kernel erosion image");
	/* get information about the preferred work-group size multiple */
	err = clGetKernelWorkGroupInfo(erosion_k, d,
		CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
		sizeof(gws_align), &gws_align, NULL);
	ocl_check(err, "preferred wg multiple for erosion");

	cl_mem d_input = NULL, d_output = NULL;

	const cl_image_format fmt = {
		.image_channel_order = CL_RGBA,
		.image_channel_data_type = CL_UNORM_INT8,
	};
	const cl_image_desc desc = {
		.image_type = CL_MEM_OBJECT_IMAGE2D,
		.image_width = imagewidth,
		.image_height = imageheight,
		//.image_row_pitch = src.data_size/src.height,
	};
	d_input = clCreateImage(ctx,
		CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
		&fmt, &desc,
		image,
		&err);
	ocl_check(err, "create image d_input");


	//SEZIONE STRUCTURING ELEMENT
	cl_mem d_strel=NULL;
	


	const cl_image_format fmt_strel = {
		.image_channel_order = CL_RGBA,
		.image_channel_data_type = CL_UNORM_INT8,
	};
	const cl_image_desc strel_desc = {
		.image_type = CL_MEM_OBJECT_IMAGE2D,
		.image_width = strelwidth,
		.image_height = strelheight,
		//.image_row_pitch = src.data_size/src.height,
	};
	d_strel = clCreateImage(ctx,
		CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
		&fmt_strel, &strel_desc,
		strel,
		&err);
	ocl_check(err, "create image d_input");

	//FINE SEZIONE STRUCTURING ELEMENT
	d_output = clCreateBuffer(ctx,
	CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
	dstdata_size, NULL,
		&err);
	ocl_check(err, "create buffer d_output");

	cl_event erosion_evt, map_evt;

	erosion_evt = erosion(erosion_k, que, d_output, d_input,d_strel, imageheight, imagewidth, strelheight,strelwidth);

	outimg = clEnqueueMapBuffer(que, d_output, CL_FALSE,
		CL_MAP_READ,
		0, dstdata_size,
		1, &erosion_evt, &map_evt, &err);
	ocl_check(err, "enqueue map d_output");

	err = clWaitForEvents(1, &map_evt);
	ocl_check(err, "clfinish");

	unsigned char * returnedArray=malloc(imagewidth*imageheight*4);

	memcpy(returnedArray,outimg,imagewidth*imageheight*4);


	err = clEnqueueUnmapMemObject(que, d_output, outimg,
		0, NULL, NULL);
	ocl_check(err, "unmap output");

	clReleaseMemObject(d_output);
	clReleaseMemObject(d_input);
	clReleaseMemObject(d_strel);
	clReleaseKernel(erosion_k);
	clReleaseProgram(prog);
	clReleaseCommandQueue(que);
	clReleaseContext(ctx);

	//stbi_image_free(image);
	//stbi_image_free(strel);
	return returnedArray;
	//TODO gestione memoria immagini in memoria dinamica
}

unsigned char * fullDilation(unsigned char* image,unsigned char* strel,int imagewidth,int imageheight,int imagechannels,int strelwidth, int strelheight,int strelchannels){

	loggingChannels(imagechannels,strelchannels);

	if(image==NULL){
		printf("error while loading the image, probably image does not exists\n");
		exit(1);
	}
	if(strel==NULL){
		printf("error while loading the image strel, probably image does not exists\n");
		exit(1);
	}
	printf("image loaded with  %i width, %i height and %i channels\n",imagewidth,imageheight,imagechannels);
	if (imagechannels <= 3) {
                fprintf(stderr, "source image must have 4 channels (<RGB,alpha> or some other format with transparency and 3 channels for color space)\n");
                exit(1);
        }
	unsigned char * outimg = NULL;
	int data_size=imagewidth*imageheight*imagechannels;
	int dstwidth=imagewidth,dstheight=imageheight;
	int dstdata_size=dstwidth*dstheight*imagechannels;
	cl_platform_id p = select_platform();
	cl_device_id d = select_device(p);
	cl_context ctx = create_context(p, d);
	cl_command_queue que = create_queue(ctx, d);
	cl_program prog = create_program("morphology.ocl", ctx, d);
	int err=0;
	cl_kernel dilation_k = NULL;
	dilation_k = clCreateKernel(prog, "dilationImage", &err);
	ocl_check(err, "create kernel erosion image");
	/* get information about the preferred work-group size multiple */
	err = clGetKernelWorkGroupInfo(dilation_k, d,
		CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
		sizeof(gws_align), &gws_align, NULL);
	ocl_check(err, "preferred wg multiple for erosion");

	cl_mem d_input = NULL, d_output = NULL;

	const cl_image_format fmt = {
		.image_channel_order = CL_RGBA,
		.image_channel_data_type = CL_UNORM_INT8,
	};
	const cl_image_desc desc = {
		.image_type = CL_MEM_OBJECT_IMAGE2D,
		.image_width = imagewidth,
		.image_height = imageheight,
		//.image_row_pitch = src.data_size/src.height,
	};
	d_input = clCreateImage(ctx,
		CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
		&fmt, &desc,
		image,
		&err);
	ocl_check(err, "create image d_input");


	//SEZIONE STRUCTURING ELEMENT
	cl_mem d_strel=NULL;
	


	const cl_image_format fmt_strel = {
		.image_channel_order = CL_RGBA,
		.image_channel_data_type = CL_UNORM_INT8,
	};
	const cl_image_desc strel_desc = {
		.image_type = CL_MEM_OBJECT_IMAGE2D,
		.image_width = strelwidth,
		.image_height = strelheight,
		//.image_row_pitch = src.data_size/src.height,
	};
	d_strel = clCreateImage(ctx,
		CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
		&fmt_strel, &strel_desc,
		strel,
		&err);
	ocl_check(err, "create image d_input");

	//FINE SEZIONE STRUCTURING ELEMENT
	d_output = clCreateBuffer(ctx,
	CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
	dstdata_size, NULL,
		&err);
	ocl_check(err, "create buffer d_output");

	cl_event erosion_evt, map_evt;

	erosion_evt = erosion(dilation_k, que, d_output, d_input,d_strel, imageheight, imagewidth, strelheight,strelwidth);

	outimg = clEnqueueMapBuffer(que, d_output, CL_FALSE,
		CL_MAP_READ,
		0, dstdata_size,
		1, &erosion_evt, &map_evt, &err);
	ocl_check(err, "enqueue map d_output");

	err = clWaitForEvents(1, &map_evt);
	ocl_check(err, "clfinish");

	unsigned char * returnedArray=malloc(imagewidth*imageheight*4);

	memcpy(returnedArray,outimg,imagewidth*imageheight*4);

	err = clEnqueueUnmapMemObject(que, d_output, outimg,
		0, NULL, NULL);
	ocl_check(err, "unmap output");

	clReleaseMemObject(d_output);
	clReleaseMemObject(d_input);
	clReleaseMemObject(d_strel);
	clReleaseKernel(dilation_k);
	clReleaseProgram(prog);
	clReleaseCommandQueue(que);
	clReleaseContext(ctx);

	//stbi_image_free(image);
	//stbi_image_free(strel);
	return returnedArray;
	//TODO gestione memoria immagini in memoria dinamica
}


unsigned char * fullGradient(unsigned char* image,unsigned char* strel,int imagewidth,int imageheight,int imagechannels,int strelwidth, int strelheight,int strelchannels){

	loggingChannels(imagechannels,strelchannels);

	if(image==NULL){
		printf("error while loading the image, probably image does not exists\n");
		exit(1);
	}
	if(strel==NULL){
		printf("error while loading the image strel, probably image does not exists\n");
		exit(1);
	}
	printf("image loaded with  %i width, %i height and %i channels\n",imagewidth,imageheight,imagechannels);
	if (imagechannels <= 3) {
                fprintf(stderr, "source image must have 4 channels (<RGB,alpha> or some other format with transparency and 3 channels for color space)\n");
                exit(1);
        }
	unsigned char * outimg = NULL;
	int data_size=imagewidth*imageheight*imagechannels;
	int dstwidth=imagewidth,dstheight=imageheight;
	int dstdata_size=dstwidth*dstheight*imagechannels;
	cl_platform_id p = select_platform();
	cl_device_id d = select_device(p);
	cl_context ctx = create_context(p, d);
	cl_command_queue que = create_queue(ctx, d);
	cl_program prog = create_program("morphology.ocl", ctx, d);
	int err=0;
	cl_kernel dilation_k = NULL,erosion_k=NULL,difference_k=NULL;
	dilation_k = clCreateKernel(prog, "dilationImage", &err);
	ocl_check(err, "create kernel dilation image");
	erosion_k = clCreateKernel(prog, "erosionImage", &err);
	ocl_check(err, "create kernel erosion image");
	difference_k = clCreateKernel(prog, "imageDifference", &err);
	ocl_check(err, "create kernel difference image");
	/* get information about the preferred work-group size multiple */
	err = clGetKernelWorkGroupInfo(dilation_k, d,
		CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
		sizeof(gws_align), &gws_align, NULL);
	ocl_check(err, "preferred wg multiple for erosion");

	cl_mem d_input = NULL, d_output = NULL;

	const cl_image_format fmt = {
		.image_channel_order = CL_RGBA,
		.image_channel_data_type = CL_UNORM_INT8,
	};
	const cl_image_desc desc = {
		.image_type = CL_MEM_OBJECT_IMAGE2D,
		.image_width = imagewidth,
		.image_height = imageheight,
		//.image_row_pitch = src.data_size/src.height,
	};
	d_input = clCreateImage(ctx,
		CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
		&fmt, &desc,
		image,
		&err);
	ocl_check(err, "create image d_input");


	//SEZIONE STRUCTURING ELEMENT
	cl_mem d_strel=NULL;
	


	const cl_image_format fmt_strel = {
		.image_channel_order = CL_RGBA,
		.image_channel_data_type = CL_UNORM_INT8,
	};
	const cl_image_desc strel_desc = {
		.image_type = CL_MEM_OBJECT_IMAGE2D,
		.image_width = strelwidth,
		.image_height = strelheight,
		//.image_row_pitch = src.data_size/src.height,
	};
	d_strel = clCreateImage(ctx,
		CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
		&fmt_strel, &strel_desc,
		strel,
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
	dilation_evt = dilation(dilation_k, que, d_output, d_input,d_strel, imageheight, imagewidth, strelheight,strelwidth);

	outimg = clEnqueueMapBuffer(que, d_output, CL_FALSE,
		CL_MAP_READ,
		0, dstdata_size,
		1, &dilation_evt, &map_evt, &err);
	ocl_check(err, "enqueue map d_output");

	err = clWaitForEvents(1, &map_evt);
	ocl_check(err, "clfinish");

	cl_mem d_tmp=NULL;

    d_tmp = clCreateImage(ctx,
		CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
		&fmt, &desc,
		outimg,
		&err);
	ocl_check(err, "create image d_tmp");

	
	// EROSION

	erosion_evt = erosion(erosion_k, que, d_output, d_input,d_strel, imageheight, imagewidth, strelheight,strelwidth);

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


	difference_evt = difference(difference_k, que, d_output, d_tmp,d_input, imageheight, imagewidth, imageheight,imagewidth);

    outimg = clEnqueueMapBuffer(que, d_output, CL_FALSE,
		CL_MAP_READ,
		0, dstdata_size,
		1, &erosion_evt, &map_evt, &err);
	ocl_check(err, "enqueue map d_output");

	err = clWaitForEvents(1, &map_evt);
	ocl_check(err, "clfinish");


	clReleaseMemObject(d_output);
	clReleaseMemObject(d_input);
	clReleaseMemObject(d_strel);
	clReleaseMemObject(d_tmp);
	clReleaseKernel(dilation_k);
	clReleaseKernel(erosion_k);
	clReleaseKernel(difference_k);
	clReleaseProgram(prog);
	clReleaseCommandQueue(que);
	clReleaseContext(ctx);

	//stbi_image_free(image);
	//stbi_image_free(strel);
	return outimg;
}

unsigned char * fullClosing(unsigned char* image,unsigned char* strel,int imagewidth,int imageheight,int imagechannels,int strelwidth, int strelheight,int strelchannels){

	loggingChannels(imagechannels,strelchannels);

	if(image==NULL){
		printf("error while loading the image, probably image does not exists\n");
		exit(1);
	}
	if(strel==NULL){
		printf("error while loading the image strel, probably image does not exists\n");
		exit(1);
	}
	printf("image loaded with  %i width, %i height and %i channels\n",imagewidth,imageheight,imagechannels);
	if (imagechannels <= 3) {
                fprintf(stderr, "source image must have 4 channels (<RGB,alpha> or some other format with transparency and 3 channels for color space)\n");
                exit(1);
        }
	unsigned char * outimg = NULL;
	int data_size=imagewidth*imageheight*imagechannels;
	int dstwidth=imagewidth,dstheight=imageheight;
	int dstdata_size=dstwidth*dstheight*imagechannels;
	cl_platform_id p = select_platform();
	cl_device_id d = select_device(p);
	cl_context ctx = create_context(p, d);
	cl_command_queue que = create_queue(ctx, d);
	cl_program prog = create_program("morphology.ocl", ctx, d);
	int err=0;
	cl_kernel erosion_k = NULL,dilation_k=NULL;
	erosion_k = clCreateKernel(prog, "erosionImage", &err);
	ocl_check(err, "create kernel erosion image");
	dilation_k = clCreateKernel(prog, "dilationImage", &err);
	ocl_check(err, "create kernel dilation image");
	/* get information about the preferred work-group size multiple */
	err = clGetKernelWorkGroupInfo(dilation_k, d,
		CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
		sizeof(gws_align), &gws_align, NULL);
	ocl_check(err, "preferred wg multiple for erosion");

	cl_mem d_input = NULL, d_output = NULL;

	const cl_image_format fmt = {
		.image_channel_order = CL_RGBA,
		.image_channel_data_type = CL_UNORM_INT8,
	};
	const cl_image_desc desc = {
		.image_type = CL_MEM_OBJECT_IMAGE2D,
		.image_width = imagewidth,
		.image_height = imageheight,
		//.image_row_pitch = src.data_size/src.height,
	};
	d_input = clCreateImage(ctx,
		CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
		&fmt, &desc,
		image,
		&err);
	ocl_check(err, "create image d_input");


	//SEZIONE STRUCTURING ELEMENT
	cl_mem d_strel=NULL;
	


	const cl_image_format fmt_strel = {
		.image_channel_order = CL_RGBA,
		.image_channel_data_type = CL_UNORM_INT8,
	};
	const cl_image_desc strel_desc = {
		.image_type = CL_MEM_OBJECT_IMAGE2D,
		.image_width = strelwidth,
		.image_height = strelheight,
		//.image_row_pitch = src.data_size/src.height,
	};
	d_strel = clCreateImage(ctx,
		CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
		&fmt_strel, &strel_desc,
		strel,
		&err);
	ocl_check(err, "create image d_input");

	//FINE SEZIONE STRUCTURING ELEMENT
	d_output = clCreateBuffer(ctx,
	CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
	dstdata_size, NULL,
		&err);
	ocl_check(err, "create buffer d_output");

	cl_event erosion_evt,dilation_evt, map_evt;

	// DILATION
	dilation_evt = dilation(dilation_k, que, d_output, d_input,d_strel, imageheight, imagewidth, strelheight,strelwidth);

	outimg = clEnqueueMapBuffer(que, d_output, CL_FALSE,
		CL_MAP_READ,
		0, dstdata_size,
		1, &dilation_evt, &map_evt, &err);
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

	// EROSION

	erosion_evt = erosion(erosion_k, que, d_output, d_input,d_strel, imageheight, imagewidth, strelheight,strelwidth);

	outimg = clEnqueueMapBuffer(que, d_output, CL_FALSE,
		CL_MAP_READ,
		0, dstdata_size,
		1, &erosion_evt, &map_evt, &err);
	ocl_check(err, "enqueue map d_output");

	err = clWaitForEvents(1, &map_evt);
	ocl_check(err, "clfinish");


	unsigned char * returnedArray=malloc(imagewidth*imageheight*4);

	memcpy(returnedArray,outimg,imagewidth*imageheight*4);

	err = clEnqueueUnmapMemObject(que, d_output, outimg,
		0, NULL, NULL);
	ocl_check(err, "unmap output");

	clReleaseMemObject(d_output);
	clReleaseMemObject(d_input);
	clReleaseMemObject(d_strel);
	clReleaseKernel(dilation_k);
	clReleaseKernel(erosion_k);
	clReleaseProgram(prog);
	clReleaseCommandQueue(que);
	clReleaseContext(ctx);

	//stbi_image_free(image);
	//stbi_image_free(strel);
	return returnedArray;
}

unsigned char * fullOpening(unsigned char* image,unsigned char* strel,int imagewidth,int imageheight,int imagechannels,int strelwidth, int strelheight,int strelchannels){

	loggingChannels(imagechannels,strelchannels);

	if(image==NULL){
		printf("error while loading the image, probably image does not exists\n");
		exit(1);
	}
	if(strel==NULL){
		printf("error while loading the image strel, probably image does not exists\n");
		exit(1);
	}
	printf("image loaded with  %i width, %i height and %i channels\n",imagewidth,imageheight,imagechannels);
	if (imagechannels <= 3) {
                fprintf(stderr, "source image must have 4 channels (<RGB,alpha> or some other format with transparency and 3 channels for color space)\n");
                exit(1);
        }
	unsigned char * outimg = NULL;
	int data_size=imagewidth*imageheight*imagechannels;
	int dstwidth=imagewidth,dstheight=imageheight;
	int dstdata_size=dstwidth*dstheight*imagechannels;
	cl_platform_id p = select_platform();
	cl_device_id d = select_device(p);
	cl_context ctx = create_context(p, d);
	cl_command_queue que = create_queue(ctx, d);
	cl_program prog = create_program("morphology.ocl", ctx, d);
	int err=0;
	cl_kernel erosion_k = NULL,dilation_k=NULL;
	erosion_k = clCreateKernel(prog, "erosionImage", &err);
	ocl_check(err, "create kernel erosion image");
	dilation_k = clCreateKernel(prog, "dilationImage", &err);
	ocl_check(err, "create kernel dilation image");
	/* get information about the preferred work-group size multiple */
	err = clGetKernelWorkGroupInfo(dilation_k, d,
		CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
		sizeof(gws_align), &gws_align, NULL);
	ocl_check(err, "preferred wg multiple for erosion");

	cl_mem d_input = NULL, d_output = NULL;

	const cl_image_format fmt = {
		.image_channel_order = CL_RGBA,
		.image_channel_data_type = CL_UNORM_INT8,
	};
	const cl_image_desc desc = {
		.image_type = CL_MEM_OBJECT_IMAGE2D,
		.image_width = imagewidth,
		.image_height = imageheight,
		//.image_row_pitch = src.data_size/src.height,
	};
	d_input = clCreateImage(ctx,
		CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
		&fmt, &desc,
		image,
		&err);
	ocl_check(err, "create image d_input");


	//SEZIONE STRUCTURING ELEMENT
	cl_mem d_strel=NULL;
	


	const cl_image_format fmt_strel = {
		.image_channel_order = CL_RGBA,
		.image_channel_data_type = CL_UNORM_INT8,
	};
	const cl_image_desc strel_desc = {
		.image_type = CL_MEM_OBJECT_IMAGE2D,
		.image_width = strelwidth,
		.image_height = strelheight,
		//.image_row_pitch = src.data_size/src.height,
	};
	d_strel = clCreateImage(ctx,
		CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
		&fmt_strel, &strel_desc,
		strel,
		&err);
	ocl_check(err, "create image d_input");

	//FINE SEZIONE STRUCTURING ELEMENT
	d_output = clCreateBuffer(ctx,
	CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
	dstdata_size, NULL,
		&err);
	ocl_check(err, "create buffer d_output");

	cl_event erosion_evt,dilation_evt, map_evt;


	//EROSION

	erosion_evt = erosion(erosion_k, que, d_output, d_input,d_strel, imageheight, imagewidth, strelheight,strelwidth);

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


	// DILATION
	dilation_evt = dilation(dilation_k, que, d_output, d_input,d_strel, imageheight, imagewidth, strelheight,strelwidth);

	outimg = clEnqueueMapBuffer(que, d_output, CL_FALSE,
		CL_MAP_READ,
		0, dstdata_size,
		1, &dilation_evt, &map_evt, &err);
	ocl_check(err, "enqueue map d_output");

	err = clWaitForEvents(1, &map_evt);
	ocl_check(err, "clfinish");

	unsigned char * returnedArray=malloc(imagewidth*imageheight*4);

	memcpy(returnedArray,outimg,imagewidth*imageheight*4);


	err = clEnqueueUnmapMemObject(que, d_output, outimg,
		0, NULL, NULL);
	ocl_check(err, "unmap output");

	clReleaseMemObject(d_output);
	clReleaseMemObject(d_input);
	clReleaseMemObject(d_strel);
	clReleaseKernel(dilation_k);
	clReleaseKernel(erosion_k);
	clReleaseProgram(prog);
	clReleaseCommandQueue(que);
	clReleaseContext(ctx);

	//stbi_image_free(image);
	//stbi_image_free(strel);
	return returnedArray;
}

unsigned char * fullTophat(unsigned char* image,unsigned char* strel,int imagewidth,int imageheight,int imagechannels,int strelwidth, int strelheight,int strelchannels){

	loggingChannels(imagechannels,strelchannels);

	if(image==NULL){
		printf("error while loading the image, probably image does not exists\n");
		exit(1);
	}
	if(strel==NULL){
		printf("error while loading the image strel, probably image does not exists\n");
		exit(1);
	}
	printf("image loaded with  %i width, %i height and %i channels\n",imagewidth,imageheight,imagechannels);
	if (imagechannels <= 3) {
                fprintf(stderr, "source image must have 4 channels (<RGB,alpha> or some other format with transparency and 3 channels for color space)\n");
                exit(1);
        }
	unsigned char * outimg = NULL;
	int data_size=imagewidth*imageheight*imagechannels;
	int dstwidth=imagewidth,dstheight=imageheight;
	int dstdata_size=dstwidth*dstheight*imagechannels;
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
	err = clGetKernelWorkGroupInfo(dilation_k, d,
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
		.image_width = imagewidth,
		.image_height = imageheight,
		//.image_row_pitch = src.data_size/src.height,
	};
	d_input = clCreateImage(ctx,
		CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
		&fmt, &desc,
		image,
		&err);
	ocl_check(err, "create image d_input");


	//SEZIONE STRUCTURING ELEMENT
	cl_mem d_strel=NULL;
	


	const cl_image_format fmt_strel = {
		.image_channel_order = CL_RGBA,
		.image_channel_data_type = CL_UNORM_INT8,
	};
	const cl_image_desc strel_desc = {
		.image_type = CL_MEM_OBJECT_IMAGE2D,
		.image_width = strelwidth,
		.image_height = strelheight,
		//.image_row_pitch = src.data_size/src.height,
	};
	d_strel = clCreateImage(ctx,
		CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
		&fmt_strel, &strel_desc,
		strel,
		&err);
	ocl_check(err, "create image d_input");

	//FINE SEZIONE STRUCTURING ELEMENT
	d_output = clCreateBuffer(ctx,
	CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
	dstdata_size, NULL,
		&err);
	ocl_check(err, "create buffer d_output");

	cl_event erosion_evt,dilation_evt, difference_evt, map_evt;


	//EROSION

	erosion_evt = erosion(erosion_k, que, d_output, d_input,d_strel, imageheight, imagewidth, strelheight,strelwidth);

	outimg = clEnqueueMapBuffer(que, d_output, CL_FALSE,
		CL_MAP_READ,
		0, dstdata_size,
		1, &erosion_evt, &map_evt, &err);
	ocl_check(err, "enqueue map d_output");
	

	err = clWaitForEvents(1, &map_evt);
	ocl_check(err, "clfinish");


	d_tmp = clCreateImage(ctx,
		CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
		&fmt, &desc,
		outimg,
		&err);
	ocl_check(err, "create image d_input after erosion");


	// DILATION
	dilation_evt = dilation(dilation_k, que, d_output, d_tmp,d_strel, imageheight, imagewidth, strelheight,strelwidth);

	outimg = clEnqueueMapBuffer(que, d_output, CL_FALSE,
		CL_MAP_READ,
		0, dstdata_size,
		1, &dilation_evt, &map_evt, &err);
	ocl_check(err, "enqueue map d_output");

	err = clWaitForEvents(1, &map_evt);
	ocl_check(err, "clfinish");

	clReleaseMemObject(d_tmp);

    d_tmp = clCreateImage(ctx,
		CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
		&fmt, &desc,
		outimg,
		&err);
	ocl_check(err, "create image d_tmp");

	//DIFFERENCE

	difference_evt = difference(difference_k, que, d_output, d_input,d_tmp, imageheight, imagewidth, imageheight,imagewidth);

    outimg = clEnqueueMapBuffer(que, d_output, CL_FALSE,
		CL_MAP_READ,
		0, dstdata_size,
		1, &erosion_evt, &map_evt, &err);
	ocl_check(err, "enqueue map d_output");

	err = clWaitForEvents(1, &map_evt);
	ocl_check(err, "clfinish");

	unsigned char * returnedArray=malloc(imagewidth*imageheight*4);

	memcpy(returnedArray,outimg,imagewidth*imageheight*4);


	err = clEnqueueUnmapMemObject(que, d_output, outimg,
		0, NULL, NULL);
	ocl_check(err, "unmap output");

	clReleaseMemObject(d_output);
	clReleaseMemObject(d_input);
	clReleaseMemObject(d_strel);
	clReleaseMemObject(d_tmp);
	clReleaseKernel(dilation_k);
	clReleaseKernel(erosion_k);
	clReleaseKernel(difference_k);
	clReleaseProgram(prog);
	clReleaseCommandQueue(que);
	clReleaseContext(ctx);

	//stbi_image_free(image);
	//stbi_image_free(strel);
	return returnedArray;
}

unsigned char * fullBottomhat(unsigned char* image,unsigned char* strel,int imagewidth,int imageheight,int imagechannels,int strelwidth, int strelheight,int strelchannels){

	loggingChannels(imagechannels,strelchannels);

	if(image==NULL){
		printf("error while loading the image, probably image does not exists\n");
		exit(1);
	}
	if(strel==NULL){
		printf("error while loading the image strel, probably image does not exists\n");
		exit(1);
	}
	printf("image loaded with  %i width, %i height and %i channels\n",imagewidth,imageheight,imagechannels);
	if (imagechannels <= 3) {
                fprintf(stderr, "source image must have 4 channels (<RGB,alpha> or some other format with transparency and 3 channels for color space)\n");
                exit(1);
        }
	unsigned char * outimg = NULL;
	int data_size=imagewidth*imageheight*imagechannels;
	int dstwidth=imagewidth,dstheight=imageheight;
	int dstdata_size=dstwidth*dstheight*imagechannels;
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
	err = clGetKernelWorkGroupInfo(dilation_k, d,
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
		.image_width = imagewidth,
		.image_height = imageheight,
		//.image_row_pitch = src.data_size/src.height,
	};
	d_input = clCreateImage(ctx,
		CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
		&fmt, &desc,
		image,
		&err);
	ocl_check(err, "create image d_input");


	//SEZIONE STRUCTURING ELEMENT
	cl_mem d_strel=NULL;
	


	const cl_image_format fmt_strel = {
		.image_channel_order = CL_RGBA,
		.image_channel_data_type = CL_UNORM_INT8,
	};
	const cl_image_desc strel_desc = {
		.image_type = CL_MEM_OBJECT_IMAGE2D,
		.image_width = strelwidth,
		.image_height = strelheight,
		//.image_row_pitch = src.data_size/src.height,
	};
	d_strel = clCreateImage(ctx,
		CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
		&fmt_strel, &strel_desc,
		strel,
		&err);
	ocl_check(err, "create image d_input");

	//FINE SEZIONE STRUCTURING ELEMENT
	d_output = clCreateBuffer(ctx,
	CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
	dstdata_size, NULL,
		&err);
	ocl_check(err, "create buffer d_output");

	cl_event erosion_evt,dilation_evt, difference_evt, map_evt;


	// DILATION
	dilation_evt = dilation(dilation_k, que, d_output, d_input,d_strel, imageheight, imagewidth, strelheight,strelwidth);

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
	ocl_check(err, "create image d_input after erosion");


	

	//EROSION

	erosion_evt = erosion(erosion_k, que, d_output, d_tmp,d_strel, imageheight, imagewidth, strelheight,strelwidth);

	outimg = clEnqueueMapBuffer(que, d_output, CL_FALSE,
		CL_MAP_READ,
		0, dstdata_size,
		1, &erosion_evt, &map_evt, &err);
	ocl_check(err, "enqueue map d_output");
	

	err = clWaitForEvents(1, &map_evt);
	ocl_check(err, "clfinish");

	clReleaseMemObject(d_tmp);

    d_tmp = clCreateImage(ctx,
		CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
		&fmt, &desc,
		outimg,
		&err);
	ocl_check(err, "create image d_tmp");

	//DIFFERENCE

	difference_evt = difference(difference_k, que, d_output, d_tmp, d_input, imageheight, imagewidth, imageheight,imagewidth);

    outimg = clEnqueueMapBuffer(que, d_output, CL_FALSE,
		CL_MAP_READ,
		0, dstdata_size,
		1, &erosion_evt, &map_evt, &err);
	ocl_check(err, "enqueue map d_output");

	err = clWaitForEvents(1, &map_evt);
	ocl_check(err, "clfinish");

	unsigned char * returnedArray=malloc(imagewidth*imageheight*4);

	memcpy(returnedArray,outimg,imagewidth*imageheight*4);


	err = clEnqueueUnmapMemObject(que, d_output, outimg,
		0, NULL, NULL);
	ocl_check(err, "unmap output");

	clReleaseMemObject(d_output);
	clReleaseMemObject(d_input);
	clReleaseMemObject(d_strel);
	clReleaseMemObject(d_tmp);
	clReleaseKernel(dilation_k);
	clReleaseKernel(erosion_k);
	clReleaseKernel(difference_k);
	clReleaseProgram(prog);
	clReleaseCommandQueue(que);
	clReleaseContext(ctx);

	//stbi_image_free(image);
	//stbi_image_free(strel);
	return returnedArray;
}

unsigned char * fullHitorMiss(unsigned char* image,unsigned char* strel,int imagewidth,int imageheight,int imagechannels,int strelwidth, int strelheight,int strelchannels){

	loggingChannels(imagechannels,strelchannels);

	if(image==NULL){
		printf("error while loading the image, probably image does not exists\n");
		exit(1);
	}
	if(strel==NULL){
		printf("error while loading the image strel, probably image does not exists\n");
		exit(1);
	}
	printf("image loaded with  %i width, %i height and %i channels\n",imagewidth,imageheight,imagechannels);
	if (imagechannels <= 3) {
                fprintf(stderr, "source image must have 4 channels (<RGB,alpha> or some other format with transparency and 3 channels for color space)\n");
                exit(1);
        }
	unsigned char * outimg = NULL;
	int data_size=imagewidth*imageheight*imagechannels;
	int dstwidth=imagewidth,dstheight=imageheight;
	int dstdata_size=dstwidth*dstheight*imagechannels;
	cl_platform_id p = select_platform();
	cl_device_id d = select_device(p);
	cl_context ctx = create_context(p, d);
	cl_command_queue que = create_queue(ctx, d);
	cl_program prog = create_program("morphology.ocl", ctx, d);
	int err=0;
	cl_kernel imageminimum_k = NULL,erosion_k=NULL,complement_k;  //TODO not dilation and not difference, but intersection-min and complement
	erosion_k = clCreateKernel(prog, "erosionImageHM", &err);
	ocl_check(err, "create kernel erosion image");
	imageminimum_k = clCreateKernel(prog, "imageMinimum", &err);
	ocl_check(err, "create kernel minimum image");
	complement_k = clCreateKernel(prog, "complement", &err);
	ocl_check(err, "create kernel complement image");
	/* get information about the preferred work-group size multiple */
	err = clGetKernelWorkGroupInfo(erosion_k, d,
		CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
		sizeof(gws_align), &gws_align, NULL);
	ocl_check(err, "preferred wg multiple for erosion");

	cl_mem d_input = NULL, d_output = NULL,d_output2=NULL;

	const cl_image_format fmt = {
		.image_channel_order = CL_RGBA,
		.image_channel_data_type = CL_UNORM_INT8,
	};
	const cl_image_desc desc = {
		.image_type = CL_MEM_OBJECT_IMAGE2D,
		.image_width = imagewidth,
		.image_height = imageheight,
		//.image_row_pitch = src.data_size/src.height,
	};
	d_input = clCreateImage(ctx,
		CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
		&fmt, &desc,
		image,
		&err);
	ocl_check(err, "create image d_input");


	//SEZIONE STRUCTURING ELEMENT
	cl_mem d_strel=NULL;
	


	const cl_image_format fmt_strel = {
		.image_channel_order = CL_RGBA,
		.image_channel_data_type = CL_UNORM_INT8,
	};
	const cl_image_desc strel_desc = {
		.image_type = CL_MEM_OBJECT_IMAGE2D,
		.image_width = strelwidth,
		.image_height = strelheight,
		//.image_row_pitch = src.data_size/src.height,
	};
	d_strel = clCreateImage(ctx,
		CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
		&fmt_strel, &strel_desc,
		strel,
		&err);
	ocl_check(err, "create image d_input");


	int streldata_size=strelheight*strelwidth*strelchannels;

	//FINE SEZIONE STRUCTURING ELEMENT


	d_output = clCreateBuffer(ctx,
	CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
	dstdata_size, NULL,
		&err);
	ocl_check(err, "create buffer d_output");

	cl_event erosion_evt,complement_evt,minimum_evt, map_evt;

	// first erosion with strel not complemented
	erosion_evt = erosion(erosion_k, que, d_output, d_input,d_strel, imageheight, imagewidth, strelheight,strelwidth);

	outimg = clEnqueueMapBuffer(que, d_output, CL_FALSE,
		CL_MAP_READ,
		0, dstdata_size,
		1, &erosion_evt, &map_evt, &err);
	ocl_check(err, "enqueue map d_output");

	err = clWaitForEvents(1, &map_evt);
	ocl_check(err, "clfinish");

	cl_mem d_tmp=NULL;

    d_tmp = clCreateImage(ctx,
		CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
		&fmt, &desc,
		outimg,
		&err);
	ocl_check(err, "create image d_tmp");



	// complement of strel


	d_output2 = clCreateBuffer(ctx,
		CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
		streldata_size, NULL,
		&err);
	ocl_check(err, "create buffer d_output2");

	complement_evt = complement(complement_k, que, d_output2,d_strel, strelheight,strelwidth);


	outimg = clEnqueueMapBuffer(que, d_output2, CL_FALSE,
		CL_MAP_READ,
		0, streldata_size,
		1, &complement_evt, &map_evt, &err);
	ocl_check(err, "enqueue map d_output2");

	err = clWaitForEvents(1, &map_evt);
	ocl_check(err, "clfinish");

	clReleaseMemObject(d_strel);

    d_strel = clCreateImage(ctx,
		CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
		&fmt_strel, &strel_desc,
		outimg,
		&err);
	ocl_check(err, "create image d_strel after complement");

	// complement of input image

	complement_evt = complement(complement_k, que, d_output,d_input, imageheight,imagewidth);


	outimg = clEnqueueMapBuffer(que, d_output, CL_FALSE,
		CL_MAP_READ,
		0, data_size,
		1, &complement_evt, &map_evt, &err);
	ocl_check(err, "enqueue map d_output");

	err = clWaitForEvents(1, &map_evt);
	ocl_check(err, "clfinish");

	clReleaseMemObject(d_input);

    d_input = clCreateImage(ctx,
		CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
		&fmt, &desc,
		outimg,
		&err);
	ocl_check(err, "create image d_input after complement");

	
	// second erosion with complemented strel

	erosion_evt = erosion(erosion_k, que, d_output, d_input,d_strel, imageheight, imagewidth, strelheight,strelwidth);

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


	minimum_evt = difference(imageminimum_k,que, d_output, d_tmp,d_input, imageheight, imagewidth, imageheight,imagewidth);

    outimg = clEnqueueMapBuffer(que, d_output, CL_FALSE,
		CL_MAP_READ,
		0, dstdata_size,
		1, &minimum_evt, &map_evt, &err);
	ocl_check(err, "enqueue map d_output");

	err = clWaitForEvents(1, &map_evt);
	ocl_check(err, "clfinish");


	clReleaseMemObject(d_output);
	clReleaseMemObject(d_output2);
	clReleaseMemObject(d_input);
	clReleaseMemObject(d_tmp);
	clReleaseMemObject(d_strel);
	clReleaseKernel(imageminimum_k);
	clReleaseKernel(erosion_k);
	clReleaseKernel(complement_k);
	clReleaseProgram(prog);
	clReleaseCommandQueue(que);
	clReleaseContext(ctx);

	//stbi_image_free(image);
	//stbi_image_free(strel);
	return outimg;
}

unsigned char* morphOperation(const char* imagename,const char* strelname,const char* method,int*finalwidth,int*finalheight,int*finalchannels){
	int width,height,channels,strelwidth,strelheight,strelchannels;
	// caricamento immagine in memoria come array di unsigned char
	unsigned char * img= stbi_load(imagename,&width,&height,&channels,STBI_rgb_alpha);

	unsigned char * imgstrel= stbi_load(strelname,&strelwidth,&strelheight,&strelchannels,STBI_rgb_alpha);

	unsigned char * processed=NULL;
	
    if(strstr(method,"erosion")){
		processed = fullErosion(img,imgstrel,width,height,channels,strelwidth,strelheight,strelchannels);
		if(processed==NULL){
			fprintf(stderr,"problems in method for erosion\n");
			exit(1);
		}
    }
    if(strstr(method,"dilation")){
        processed = fullDilation(img,imgstrel,width,height,channels,strelwidth,strelheight,strelchannels);
		if(processed==NULL){
			fprintf(stderr,"problems in method for dilation\n");
			exit(1);
		}
    }
    if(strstr(method,"gradient")){
        processed = fullGradient(img,imgstrel,width,height,channels,strelwidth,strelheight,strelchannels);
		if(processed==NULL){
			fprintf(stderr,"problems in method for gradient\n");
			exit(1);
		}
    }
    if(strstr(method,"opening")){
        processed = fullOpening(img,imgstrel,width,height,channels,strelwidth,strelheight,strelchannels);
		if(processed==NULL){
			fprintf(stderr,"problems in method for opening\n");
			exit(1);
		}
    }
    if(strstr(method,"closing")){
        processed = fullClosing(img,imgstrel,width,height,channels,strelwidth,strelheight,strelchannels);
		if(processed==NULL){
			fprintf(stderr,"problems in method for closing\n");
			exit(1);
		}
    }
    if(strstr(method,"tophat")){
        processed = fullTophat(img,imgstrel,width,height,channels,strelwidth,strelheight,strelchannels);
		if(processed==NULL){
			fprintf(stderr,"problems in method for tophat\n");
			exit(1);
		}
    }
    if(strstr(method,"bottomhat")){
        processed = fullBottomhat(img,imgstrel,width,height,channels,strelwidth,strelheight,strelchannels);
		if(processed==NULL){
			fprintf(stderr,"problems in method for bottomhat\n");
			exit(1);
		}
    }

	if(strstr(method,"hitormiss")){
        processed = fullHitorMiss(img,imgstrel,width,height,channels,strelwidth,strelheight,strelchannels);
		if(processed==NULL){
			fprintf(stderr,"problems in method for hitormiss\n");
			exit(1);
		}
    }

	if(!processed){
		printf("the operation %s is not implemented\n",method);
	}

	stbi_image_free(img);
	stbi_image_free(imgstrel);

	*finalwidth=width;
	*finalheight=height;
	*finalchannels=channels;

	return processed;

}