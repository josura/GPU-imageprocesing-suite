#include<stdio.h>
#include<string.h>

#define STB_IMAGE_IMPLEMENTATION
#include"../../../stb/stb_image.h" 
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include"../../../stb/stb_image_write.h" 


#define CL_TARGET_OPENCL_VERSION 120
#include "../ocl_boiler.h"


size_t gws_align_imgzoom;

cl_event imgzoom(cl_kernel imgzoom_k, cl_command_queue que,
	cl_mem d_output, cl_mem d_input,
	cl_int nrows, cl_int ncols)
{
	const size_t gws[] = { round_mul_up(ncols, gws_align_imgzoom), nrows };
	cl_event imgzoom_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(imgzoom_k, i++, sizeof(d_output), &d_output);
	ocl_check(err, "set imgzoom arg", i-1);
	err = clSetKernelArg(imgzoom_k, i++, sizeof(d_input), &d_input);
	ocl_check(err, "set imgzoom arg", i-1);

	err = clEnqueueNDRangeKernel(que, imgzoom_k, 2,
		NULL, gws, NULL,
		0, NULL, &imgzoom_evt);

	ocl_check(err, "enqueue imgzoom");

	return imgzoom_evt;
}


cl_event transpose(cl_kernel transpose_k, const size_t *lws,
        cl_command_queue que,
        cl_mem d_T, cl_mem d_I, cl_int nrows_T, cl_int ncols_T, cl_event init_evt)
{
        const size_t gws[] = { round_mul_up(ncols_T, lws[0]), round_mul_up(nrows_T, lws[1]) };
        cl_event trans_evt;
        cl_int err;

        cl_uint i = 0;
        err = clSetKernelArg(transpose_k, i++, sizeof(d_T), &d_T);
        ocl_check(err, "set transpose arg", i-1);
        err = clSetKernelArg(transpose_k, i++, sizeof(d_I), &d_I);
        ocl_check(err, "set transpose arg", i-1);

        err = clEnqueueNDRangeKernel(que, transpose_k, 2,
                NULL, gws, lws,
                1, &init_evt, &trans_evt);

        ocl_check(err, "enqueue tranpose");

        return trans_evt;
}





void usage(int argc){
	if(argc<2){
		fprintf(stderr,"Usage: ./load_image <image.png>");
		exit(1);
	}
}

int main(int argc, char ** args){
	usage(argc);
	int width,height,channels;
	// caricamento immagine in memoria come array di unsigned char
	unsigned char * img= stbi_load(args[1],&width,&height,&channels,STBI_rgb_alpha);
	if(img==NULL){
		printf("errore nel caricamento dell'immagine");
	}
	printf("immagine caricata con larghezza %i, altezza %i e canali %i\n",width,height,channels);
	if (channels < 3) {
                fprintf(stderr, "source must have 4 channels\n");
                exit(1);
        }
	unsigned char * outimg = NULL;
	int data_size=width*height*channels;
	int dstwidth=width*2,dstheight=height*2;
	int dstdata_size=dstwidth*dstheight*channels;
	cl_platform_id p = select_platform();
	cl_device_id d = select_device(p);
	cl_context ctx = create_context(p, d);
	cl_command_queue que = create_queue(ctx, d);
	cl_program prog = create_program("imgzoom.ocl", ctx, d);
	int err=0;
	cl_kernel imgzoom_k = clCreateKernel(prog, "imgzoom", &err);
	ocl_check(err, "create kernel imgzoom");

	/* get information about the preferred work-group size multiple */
	err = clGetKernelWorkGroupInfo(imgzoom_k, d,
		CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
		sizeof(gws_align_imgzoom), &gws_align_imgzoom, NULL);
	ocl_check(err, "preferred wg multiple for imgzoom");

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

	cl_event imgzoom_evt, map_evt;

	imgzoom_evt = imgzoom(imgzoom_k, que, d_output, d_input, height, width);

	outimg = clEnqueueMapBuffer(que, d_output, CL_FALSE,
		CL_MAP_READ,
		0, dstdata_size,
		1, &imgzoom_evt, &map_evt, &err);
	ocl_check(err, "enqueue map d_output");

	err = clWaitForEvents(1, &map_evt);
	ocl_check(err, "clfinish");

	const double runtime_imgzoom_ms = runtime_ms(imgzoom_evt);
	const double runtime_map_ms = runtime_ms(map_evt);

	const double imgzoom_bw_gbs = 2.0*dstdata_size/1.0e6/runtime_imgzoom_ms;
	const double map_bw_gbs = dstdata_size/1.0e6/runtime_map_ms;

	printf("imgzoom: %dx%d int in %gms: %g GB/s %g GE/s\n",
		height, width, runtime_imgzoom_ms, imgzoom_bw_gbs, height*width/1.0e6/runtime_imgzoom_ms);
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
	clReleaseKernel(imgzoom_k);
	clReleaseProgram(prog);
	clReleaseCommandQueue(que);
	clReleaseContext(ctx);








	stbi_image_free(img);
	
	return 0;
}
