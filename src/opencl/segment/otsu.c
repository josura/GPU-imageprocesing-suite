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


size_t gws_align_histogram, gws_align_binarization;

//Get one of the seeds
static inline uint64_t rdtsc(void)
{
	uint64_t val;
	uint32_t h, l;
    __asm__ __volatile__("rdtsc" : "=a" (l), "=d" (h));
        val = ((uint64_t)l) | (((uint64_t)h) << 32);
        return val;
}

cl_event histogram(cl_kernel histogram_k, cl_command_queue que, cl_mem d_input,
	cl_mem d_histogram, cl_int nrows, cl_int ncols)
{
	const size_t gws[] = { round_mul_up(ncols, gws_align_histogram), nrows };
	cl_event histogram_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(histogram_k, i++, sizeof(d_histogram), &d_histogram);
	ocl_check(err, "set histogram arg %d", i-1);
	err = clSetKernelArg(histogram_k, i++, sizeof(d_input), &d_input);
	ocl_check(err, "set histogram arg %d", i-1);

	err = clEnqueueNDRangeKernel(que, histogram_k, 2,
		NULL, gws, NULL,
		0, NULL, &histogram_evt);

	ocl_check(err, "enqueue histogram");

	return histogram_evt;
}

cl_event scanN(cl_kernel scanN_k, cl_command_queue que,
	cl_mem d_vsum, cl_mem d_tails, cl_mem d_v1, cl_int nels,
	cl_int lws_, cl_int nwg, float n_pixels,
	cl_event init_evt)
{
	const size_t gws[] = { nwg*lws_ };
	const size_t lws[] = { lws_ };
	cl_event scanN_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(scanN_k, i++, sizeof(d_vsum), &d_vsum);
	ocl_check(err, "set scanN arg", i-1);
	err = clSetKernelArg(scanN_k, i++, sizeof(d_tails), &d_tails);
	ocl_check(err, "set scanN arg %d", i-1);
	err = clSetKernelArg(scanN_k, i++, sizeof(d_v1), &d_v1);
	ocl_check(err, "set scanN arg %d", i-1);
	err = clSetKernelArg(scanN_k, i++, sizeof(float)*2*lws[0], NULL);
	ocl_check(err, "set scanN arg %d", i-1);
	err = clSetKernelArg(scanN_k, i++, sizeof(nels), &nels);
	ocl_check(err, "set scanN arg %d", i-1);
	err = clSetKernelArg(scanN_k, i++, sizeof(n_pixels), &n_pixels);
	ocl_check(err, "set scanN arg %d", i-1);

	err = clEnqueueNDRangeKernel(que, scanN_k, 1,
		NULL, gws, lws,
		1, &init_evt, &scanN_evt);

	ocl_check(err, "enqueue scanN_lmem");

	err = clFinish(que);

	ocl_check(err, "finish que");


	return scanN_evt;
}

cl_event scan1(cl_kernel scan1_k, cl_command_queue que,
	cl_mem d_vsum, cl_mem d_tails, cl_mem d_v1, cl_int nels,
	cl_int lws_, cl_int nwg,
	cl_event init_evt)
{
	const size_t gws[] = { nwg*lws_ };
	const size_t lws[] = { lws_ };
	cl_event scan1_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(scan1_k, i++, sizeof(d_vsum), &d_vsum);
	ocl_check(err, "set scan1 arg", i-1);
	if (nwg > 1) {
		err = clSetKernelArg(scan1_k, i++, sizeof(d_tails), &d_tails);
		ocl_check(err, "set scan1 arg %d", i-1);
	}
	err = clSetKernelArg(scan1_k, i++, sizeof(d_v1), &d_v1);
	ocl_check(err, "set scan1 arg %d", i-1);
	err = clSetKernelArg(scan1_k, i++, sizeof(float)*2*lws[0], NULL);
	ocl_check(err, "set scan1 arg %d", i-1);
	err = clSetKernelArg(scan1_k, i++, sizeof(nels), &nels);
	ocl_check(err, "set scan1 arg %d", i-1);

	err = clEnqueueNDRangeKernel(que, scan1_k, 1,
		NULL, gws, lws,
		1, &init_evt, &scan1_evt);

	ocl_check(err, "enqueue scan_lmem");

	err = clFinish(que);

	ocl_check(err, "finish que");


	return scan1_evt;
}

cl_event fixup(cl_kernel fixup_k, cl_command_queue que,
	cl_mem d_vsum, cl_mem d_tails, cl_int nels,
	cl_int lws_, cl_int nwg,
	cl_event init_evt)
{
	const size_t gws[] = { nwg*lws_ };
	const size_t lws[] = { lws_ };
	cl_event fixup_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(fixup_k, i++, sizeof(d_vsum), &d_vsum);
	ocl_check(err, "set fixup arg %d", i-1);
	err = clSetKernelArg(fixup_k, i++, sizeof(d_tails), &d_tails);
	ocl_check(err, "set fixup arg %d", i-1);
	err = clSetKernelArg(fixup_k, i++, sizeof(nels), &nels);
	ocl_check(err, "set fixup arg %d", i-1);

	err = clEnqueueNDRangeKernel(que, fixup_k, 1,
		NULL, gws, lws,
		1, &init_evt, &fixup_evt);

	ocl_check(err, "enqueue scan_lmem");

	err = clFinish(que);

	ocl_check(err, "finish que");


	return fixup_evt;
}

cl_event otsu(cl_kernel otsu_k, cl_command_queue que, int _gws, int _lws,
	cl_mem d_probs_means, cl_mem d_max_wg_k, float g_mean)
{
	const size_t gws[] = { _gws };
	const size_t lws[] = { _lws };
	cl_event otsu_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(otsu_k, i++, sizeof(d_probs_means), &d_probs_means);
	ocl_check(err, "set otsu arg %d", i-1);
	err = clSetKernelArg(otsu_k, i++, sizeof(d_max_wg_k), &d_max_wg_k);
	ocl_check(err, "set otsu arg %d", i-1);
	err = clSetKernelArg(otsu_k, i++, sizeof(float)*2*lws[0], NULL);
	ocl_check(err, "set otsu arg %d", i-1);
	err = clSetKernelArg(otsu_k, i++, sizeof(g_mean), &g_mean);
	ocl_check(err, "set otsu arg %d", i-1);

	err = clEnqueueNDRangeKernel(que, otsu_k, 1,
		NULL, gws, lws,
		0, NULL, &otsu_evt);

	ocl_check(err, "enqueue otsu");

	return otsu_evt;
}

cl_event reducemax(cl_kernel reducemax_k, cl_command_queue que, int _gws, int _lws,
	cl_mem d_max_wg_k, cl_mem d_max_k, cl_event otsu_evt)
{
	const size_t gws[] = { _gws };
	const size_t lws[] = { _lws };
	cl_event reducemax_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(reducemax_k, i++, sizeof(d_max_wg_k), &d_max_wg_k);
	ocl_check(err, "set reducemax arg %d", i-1);
	err = clSetKernelArg(reducemax_k, i++, sizeof(d_max_k), &d_max_k);
	ocl_check(err, "set reducemax arg %d", i-1);
	err = clSetKernelArg(reducemax_k, i++, sizeof(float)*2*lws[0], NULL);
	ocl_check(err, "set reducemax arg %d", i-1);

	err = clEnqueueNDRangeKernel(que, reducemax_k, 1,
		NULL, gws, lws,
		1, &otsu_evt, &reducemax_evt);

	ocl_check(err, "enqueue reducemax");

	return reducemax_evt;
}


cl_event binarization(cl_kernel binarization_k, cl_command_queue que,
	cl_mem d_output, cl_mem d_input,
	cl_int threshold, cl_int nrows, cl_int ncols)
{
	const size_t gws[] = { round_mul_up(ncols, gws_align_binarization), nrows };
	cl_event binarization_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(binarization_k, i++, sizeof(d_output), &d_output);
	ocl_check(err, "set binarization arg %d", i-1);
	err = clSetKernelArg(binarization_k, i++, sizeof(d_input), &d_input);
	ocl_check(err, "set binarization arg %d", i-1);
	err = clSetKernelArg(binarization_k, i++, sizeof(threshold), &threshold);
	ocl_check(err, "set binarization arg %d", i-1);

	err = clEnqueueNDRangeKernel(que, binarization_k, 2,
		NULL, gws, NULL,
		0, NULL, &binarization_evt);

	ocl_check(err, "enqueue binarization");

	return binarization_evt;
}

void usage(int argc){
	if(argc<3){
		fprintf(stderr,"Usage: ./otsu <image.png> <output.png>\n ");
		fprintf(stderr,"The image needs to have 4 channels (R,G,B,transparency)\n");
		fprintf(stderr,"It will be converted to grayscale with 1 channel\n");
		fprintf(stderr,"output is the name of the output image\n");

		exit(1);
	}
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

void host_histogram(cl_int * hist, cl_uchar4 * img, int width, int height){
	int level, img_index;
	for(int i=0; i<width; ++i){
		for(int j=0; j<height; ++j){
			img_index = j*height+i;
			level = img[img_index].x;
			hist[level]++;
		}
	}
}

//Those values are cumulative, it would be counterproductive using the GPU
void host_probs_means(int * histogram, int n_pixels, float * probs, float * cumulative_means){
	float * freq = malloc(sizeof(float)*255);
	//Compute frequencies
	for(int i=0; i<256; ++i){
		freq[i] = ((float)histogram[i])/n_pixels;
	}
	//Compute class probabilities and cumulative means
	float sum_prob = 0, sum_mean = 0;
	for(int i=0; i<256; ++i){
		sum_prob += freq[i];
		sum_mean += i*freq[i];
		probs[i] = sum_prob;
		cumulative_means[i] = sum_mean;
	}
	free(freq);
}

int main(int argc, char ** args){
	usage(argc);
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
	cl_kernel histogram_k = clCreateKernel(prog, "histogram", &err);
	ocl_check(err, "create kernel histogram");
	cl_kernel scan1_k = clCreateKernel(prog, "scan1_lmem", &err);
	ocl_check(err, "create kernel scan1_lmem");
	cl_kernel scanN_k = clCreateKernel(prog, "scanN_lmem", &err);
	ocl_check(err, "create kernel scanN_lmem");
	cl_kernel fixup_k = clCreateKernel(prog, "scanN_fixup", &err);
	ocl_check(err, "create kernel scanN_fixup");
	cl_kernel otsu_k = clCreateKernel(prog, "otsu", &err);
	ocl_check(err, "create kernel otsu");
	cl_kernel reducemax_k = clCreateKernel(prog, "reducemax", &err);
	ocl_check(err, "create kernel reducemax");
	cl_kernel binarization_k = clCreateKernel(prog, "binarization", &err);
	ocl_check(err, "create kernel binarization");
    
	/* get information about the preferred work-group size multiple */
	err = clGetKernelWorkGroupInfo(histogram_k, d,
		CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
		sizeof(gws_align_histogram), &gws_align_histogram, NULL);
	ocl_check(err, "preferred wg multiple for histogram");
	err = clGetKernelWorkGroupInfo(binarization_k, d,
		CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
		sizeof(gws_align_binarization), &gws_align_binarization, NULL);
	ocl_check(err, "preferred wg multiple for binarization");

	cl_int lws = gws_align_histogram;
	cl_int nwg = 256/lws;

	cl_mem d_input = NULL, d_output = NULL, d_histogram = NULL, d_tails = NULL, d_probs_means = NULL, d_max_wg_k = NULL, d_max_k = NULL;

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

	d_output = clCreateBuffer(ctx,
	CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
	dstdata_size, NULL,
		&err);
	ocl_check(err, "create buffer d_output");

	d_histogram = clCreateBuffer(ctx,
	CL_MEM_READ_WRITE,
	sizeof(cl_int)*256, NULL,
		&err);
	ocl_check(err, "create buffer d_histogram");

	d_max_wg_k = clCreateBuffer(ctx,
	CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY,
	sizeof(float)*2*8, NULL,
		&err);
	ocl_check(err, "create buffer d_max_wg_k");
	
	d_max_k = clCreateBuffer(ctx,
	CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
	sizeof(cl_int), NULL,
		&err);
	ocl_check(err, "create buffer d_max_k");

	cl_event histogram_evt, map_hist_evt, scan_evt[3], otsu_evt, reducemax_evt, map_k_evt, binarization_evt, map_output_evt;
	
	histogram_evt = histogram(histogram_k, que, d_input, d_histogram, height, width);
	
	d_probs_means = clCreateBuffer(ctx,
	CL_MEM_READ_WRITE,
	sizeof(float)*256*2, NULL,
		&err);
	ocl_check(err, "create buffer d_probs_means");
	d_tails = clCreateBuffer(ctx,
	CL_MEM_READ_WRITE,
	sizeof(float)*2*nwg, NULL,
		&err);
	ocl_check(err, "create buffer d_tails");

	float n_pixels = (float)(width*height);

	// riduco il datasize originale ad nwg elementi
	scan_evt[0] = scanN(scanN_k, que, d_probs_means, d_tails, d_histogram, 256,
		lws, nwg, n_pixels, histogram_evt);
	if (nwg > 1) {
		scan_evt[1] = scan1(scan1_k, que, d_tails, NULL, d_tails, nwg,
			lws, 1, scan_evt[0]);
		scan_evt[2] = fixup(fixup_k, que, d_probs_means, d_tails, 256, lws, nwg,
			scan_evt[1]);
	} else {
		scan_evt[2] = scan_evt[1] = scan_evt[0];
	}

	float g_mean;
	//Read g_mean in d_probs_means[255].y
	clEnqueueReadBuffer(que, d_probs_means, CL_TRUE, sizeof(float)*2*255+sizeof(float), sizeof(float), &g_mean, 1, scan_evt+2, NULL);
	printf("Global mean is %f\n", g_mean);
	
	otsu_evt = otsu(otsu_k, que, 256, 32, d_probs_means, d_max_wg_k, g_mean);
	reducemax_evt = reducemax(reducemax_k, que, 8, 8, d_max_wg_k, d_max_k, otsu_evt);

	cl_int * threshold = clEnqueueMapBuffer(que, d_max_k, CL_TRUE,
		CL_MAP_READ,
		0, sizeof(cl_int),
		1, &reducemax_evt, &map_k_evt, &err);
	ocl_check(err, "enqueue map d_max_k");
	printf("Ideal threshold computed with Otsu on GPU is %d\n", *threshold);

	binarization_evt = binarization(binarization_k, que, d_output, d_input, *threshold, height, width);	
	outimg = clEnqueueMapBuffer(que, d_output, CL_TRUE,
		CL_MAP_READ,
		0, dstdata_size,
		1, &binarization_evt, &map_output_evt, &err);
	ocl_check(err, "enqueue map d_output");

	const double runtime_histogram_ms = runtime_ms(histogram_evt);
	const double runtime_scan_ms = total_runtime_ms(scan_evt[0], scan_evt[2]);
	const double runtime_otsu_ms = total_runtime_ms(otsu_evt, reducemax_evt);
	const double runtime_binarization_ms = runtime_ms(binarization_evt);
	const double runtime_map_ms = runtime_ms(map_output_evt);
	const double total_runtime_ms = runtime_histogram_ms+runtime_scan_ms+runtime_otsu_ms+runtime_binarization_ms+runtime_map_ms;

	const double histogram_bw_gbs = dstdata_size/1.0e6/runtime_histogram_ms;
	const double scan_bw_gbs = sizeof(float)*256*2/1.0e6/runtime_scan_ms;
	const double otsu_bw_gbs = sizeof(float)*256*2/1.0e6/runtime_otsu_ms;
	const double binarization_bw_gbs = dstdata_size/1.0e6/runtime_binarization_ms;
	const double map_bw_gbs = dstdata_size/1.0e6/runtime_map_ms;

	printf("histogram: %dx%d uint in %gms: %g GB/s %g GE/s\n",
		height, width, runtime_histogram_ms, histogram_bw_gbs, height*width/1.0e6/runtime_histogram_ms);
	printf("scan : 256 probabilities and 256 means computed in %gms: %g GE/s\n",
		runtime_scan_ms, 256*2/1.0e6/runtime_scan_ms);
	printf("otsu: 256 interclass variances computed and reduced in %gms: %g GB/s %g GE/s\n",
		runtime_otsu_ms, otsu_bw_gbs, 256/1.0e6/runtime_otsu_ms);
	printf("binarization: %dx%d uint in %gms: %g GB/s %g GE/s\n",
		height, width, runtime_binarization_ms, binarization_bw_gbs, height*width/1.0e6/runtime_binarization_ms);
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
	clReleaseMemObject(d_histogram);
	clReleaseMemObject(d_max_k);
	clReleaseMemObject(d_max_wg_k);
	clReleaseMemObject(d_probs_means);
	clReleaseMemObject(d_tails);
	clReleaseKernel(histogram_k);
	clReleaseKernel(otsu_k);
	clReleaseKernel(binarization_k);
	clReleaseProgram(prog);
	clReleaseCommandQueue(que);
	clReleaseContext(ctx);

	stbi_image_free(img);
	return 0;
}
