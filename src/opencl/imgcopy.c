#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define CL_TARGET_OPENCL_VERSION 120
#include "ocl_boiler.h"

#include "pamalign.h"

size_t gws_align_imgcopy;

cl_event imgcopy(cl_kernel imgcopy_k, cl_command_queue que,
	cl_mem d_output, cl_mem d_input,
	cl_int nrows, cl_int ncols)
{
	const size_t gws[] = { round_mul_up(ncols, gws_align_imgcopy), nrows };
	cl_event imgcopy_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(imgcopy_k, i++, sizeof(d_output), &d_output);
	ocl_check(err, "set imgcopy arg", i-1);
	err = clSetKernelArg(imgcopy_k, i++, sizeof(d_input), &d_input);
	ocl_check(err, "set imgcopy arg", i-1);

	err = clEnqueueNDRangeKernel(que, imgcopy_k, 2,
		NULL, gws, NULL,
		0, NULL, &imgcopy_evt);

	ocl_check(err, "enqueue imgcopy");

	return imgcopy_evt;
}

int main(int argc, char *argv[])
{
	if (argc <= 1) {
		fprintf(stderr, "specify input file\n");
		exit(1);
	}

	const char *input_fname = argv[1];
	const char *output_fname = "copia.pam";

	struct imgInfo src;
	struct imgInfo dst;
	cl_int err = load_pam(input_fname, &src);
	if (err != 0) {
		fprintf(stderr, "error loading %s\n", input_fname);
		exit(1);
	}
	if (src.channels != 4) {
		fprintf(stderr, "source must have 4 channels\n");
		exit(1);
	}
	if (src.depth != 8) {
		fprintf(stderr, "source must have 8-bit channels\n");
		exit(1);
	}
	dst = src;
	dst.data = NULL;

	cl_platform_id p = select_platform();
	cl_device_id d = select_device(p);
	cl_context ctx = create_context(p, d);
	cl_command_queue que = create_queue(ctx, d);
	cl_program prog = create_program("imgcopy.ocl", ctx, d);

	cl_kernel imgcopy_k = clCreateKernel(prog, "imgcopy", &err);
	ocl_check(err, "create kernel imgcopy");

	/* get information about the preferred work-group size multiple */
	err = clGetKernelWorkGroupInfo(imgcopy_k, d,
		CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
		sizeof(gws_align_imgcopy), &gws_align_imgcopy, NULL);
	ocl_check(err, "preferred wg multiple for imgcopy");

	cl_mem d_input = NULL, d_output = NULL;

	const cl_image_format fmt = {
		.image_channel_order = CL_RGBA,
		.image_channel_data_type = CL_UNORM_INT8,
	};
	const cl_image_desc desc = {
		.image_type = CL_MEM_OBJECT_IMAGE2D,
		.image_width = src.width,
		.image_height = src.height,
		//.image_row_pitch = src.data_size/src.height,
	};
	d_input = clCreateImage(ctx,
		CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
		&fmt, &desc,
		src.data,
		&err);
	ocl_check(err, "create image d_input");
	d_output = clCreateBuffer(ctx,
		CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
		dst.data_size, NULL,
		&err);
	ocl_check(err, "create buffer d_output");

	cl_event imgcopy_evt, map_evt;

	imgcopy_evt = imgcopy(imgcopy_k, que, d_output, d_input, src.height, src.width);

	dst.data = clEnqueueMapBuffer(que, d_output, CL_FALSE,
		CL_MAP_READ,
		0, dst.data_size,
		1, &imgcopy_evt, &map_evt, &err);
	ocl_check(err, "enqueue map d_output");

	err = clWaitForEvents(1, &map_evt);
	ocl_check(err, "clfinish");

	const double runtime_imgcopy_ms = runtime_ms(imgcopy_evt);
	const double runtime_map_ms = runtime_ms(map_evt);

	const double imgcopy_bw_gbs = 1.0*src.data_size/1.0e6/runtime_imgcopy_ms;
	const double map_bw_gbs = src.data_size/1.0e6/runtime_map_ms;

	printf("imgcopy: %dx%d int in %gms: %g GB/s %g GE/s\n",
		src.height, src.width, runtime_imgcopy_ms, imgcopy_bw_gbs, src.height*src.width/1.0e6/runtime_imgcopy_ms);
	printf("map: %dx%d int in %gms: %g GB/s %g GE/s\n",
		src.height, src.width, runtime_map_ms, map_bw_gbs, src.height*src.width/1.0e6/runtime_map_ms);

	err = save_pam(output_fname, &dst);
	if (err != 0) {
		fprintf(stderr, "error writing %s\n", output_fname);
		exit(1);
	}

	err = clEnqueueUnmapMemObject(que, d_output, dst.data,
		0, NULL, NULL);
	ocl_check(err, "unmap output");

	clReleaseMemObject(d_output);
	clReleaseMemObject(d_input);
	clReleaseKernel(imgcopy_k);
	clReleaseProgram(prog);
	clReleaseCommandQueue(que);
	clReleaseContext(ctx);

	return 0;

}
