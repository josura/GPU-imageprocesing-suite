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


size_t gws_align_binarization;

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
		fprintf(stderr,"Usage: ./otsu_cpu <image.png> <output.png>\n ");
		fprintf(stderr,"The image can have 1 to 4 channels (R,G,B,transparency)\n");
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

void host_probs_means(int * histogram, int n_pixels, float * probs, float * cumulative_means){
	float * freq = malloc(sizeof(float)*255);
	//Compute frequencies
	for(int i=0; i<256; ++i){
		freq[i] = ((float)histogram[i])/n_pixels;
		//printf("Hist %d: %d\n", i, histogram[i]);
		//printf("Freq %d: %f\n", i, freq[i]);
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

int host_otsu_alternative(float * probs, float * cumulative_means, int g_mean){
	float var, prob1, prob2, cumulative_mean1, mean1, mean2;
	int max_index = 0;
	float curr_var = 0;	
	for(int i=1; i<255; ++i){
		prob1 = probs[i];
		prob2 = 1.0f - prob1;
		cumulative_mean1 = cumulative_means[i];
		mean1 = cumulative_mean1/prob1;
		mean2 = (g_mean - cumulative_mean1)/prob2;
		var = prob1*prob2*pow(mean1-mean2, 2);
		//printf("CPU i %d, var: %f\n", i, var);
		if (var > curr_var){
			max_index = i;
			curr_var = var;
		}
	}
	return max_index;
}

int host_otsu(float * probs, float * cumulative_means, int g_mean){
	float num, var;
	int max_index = 0;
	float curr_var = 0;
	float log_num, log_probs, log_probs2, log_var;
	for(int i=1; i<255; ++i){
		num = ((g_mean*probs[i])-cumulative_means[i]);
		log_num = log(num);
		log_probs = log(probs[i]);
		log_probs2 = log(1-probs[i]);
		log_var = 2*log_num-log_probs-log_probs2;
		var = exp(log_var);
		if (var > curr_var){
			max_index = i;
			curr_var = var;
		}
	}
	return max_index;
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
	cl_kernel binarization_k = clCreateKernel(prog, "binarization", &err);
	ocl_check(err, "create kernel binarization");
    
	/* get information about the preferred work-group size multiple */
	err = clGetKernelWorkGroupInfo(binarization_k, d,
		CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
		sizeof(gws_align_binarization), &gws_align_binarization, NULL);
	ocl_check(err, "preferred wg multiple for binarization");

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

	d_output = clCreateBuffer(ctx,
	CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
	dstdata_size, NULL,
		&err);
	ocl_check(err, "create buffer d_output");

	clock_t start_hist, end_hist;
	clock_t start_otsu, end_otsu;

	start_hist = clock();
	int * hist = calloc(256, sizeof(cl_int));
	host_histogram(hist, (cl_uchar4*)img, width, height);
	end_hist = clock();

	start_otsu = clock();
	float * probs = malloc(sizeof(float)*256);
	float * cumulative_means = malloc(sizeof(float)*256);
	host_probs_means(hist, height*width, probs, cumulative_means);
	
	float g_mean = cumulative_means[255];
	
	int host_threshold = host_otsu(probs, cumulative_means, g_mean);
	end_otsu = clock();

	printf("Global mean is %f\n", g_mean);
	printf("Ideal threshold computed with Otsu on CPU is %d\n", host_threshold);

	cl_event binarization_evt = binarization(binarization_k, que, d_output, d_input, host_threshold, height, width);	
	cl_event map_output_evt;
	outimg = clEnqueueMapBuffer(que, d_output, CL_TRUE,
		CL_MAP_READ,
		0, dstdata_size,
		1, &binarization_evt, &map_output_evt, &err);
	ocl_check(err, "enqueue map d_output");

	const double runtime_histogram_ms = (end_hist - start_hist)*1.0e3/CLOCKS_PER_SEC;
	const double runtime_otsu_ms = (end_otsu - start_otsu)*1.0e3/CLOCKS_PER_SEC;
	const double runtime_binarization_ms = runtime_ms(binarization_evt);
	const double runtime_map_ms = runtime_ms(map_output_evt);

	const double histogram_bw_gbs = dstdata_size/1.0e6/runtime_histogram_ms;
	const double otsu_bw_gbs = sizeof(float)*256*2/1.0e6/runtime_otsu_ms;
	const double binarization_bw_gbs = dstdata_size/1.0e6/runtime_binarization_ms;
	const double map_bw_gbs = dstdata_size/1.0e6/runtime_map_ms;

	printf("histogram: %dx%d uint in %gms: %g GB/s %g GE/s\n",
		height, width, runtime_histogram_ms, histogram_bw_gbs, height*width/1.0e6/runtime_histogram_ms);
	printf("otsu: 256 interclass variances computed and reduced in %gms: %g GB/s %g GE/s\n",
		runtime_otsu_ms, otsu_bw_gbs, 256/1.0e6/runtime_otsu_ms);
	printf("binarization: %dx%d uint in %gms: %g GB/s %g GE/s\n",
		height, width, runtime_binarization_ms, binarization_bw_gbs, height*width/1.0e6/runtime_binarization_ms);
	printf("map: %dx%d int in %gms: %g GB/s %g GE/s\n",
		dstheight, dstwidth, runtime_map_ms, map_bw_gbs, dstheight*dstwidth/1.0e6/runtime_map_ms);

	char outputImage[128];
	sprintf(outputImage,"%s",args[2]);
	printf("image saved as %s\n",outputImage);
	stbi_write_png(outputImage,dstwidth,dstheight,channels,outimg,channels*dstwidth);

	err = clEnqueueUnmapMemObject(que, d_output, outimg,
		0, NULL, NULL);
	ocl_check(err, "unmap output");

	clReleaseMemObject(d_output);
	clReleaseMemObject(d_input);
	clReleaseKernel(binarization_k);
	clReleaseProgram(prog);
	clReleaseCommandQueue(que);
	clReleaseContext(ctx);

	stbi_image_free(img);
	free(hist);
	free(probs);
	free(cumulative_means);
	return 0;
}
