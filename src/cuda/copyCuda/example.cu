#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <stdarg.h>
#include <cuda_runtime_api.h>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>


using namespace std;


/* Check a CUDA error status, printing a message and exiting
 * in case of failure
 */
#define BUFSIZE 4096
void cuda_check(cudaError_t err, const char *msg, ...) {
        if (err != cudaSuccess) {
                char msg_buf[BUFSIZE + 1];
                va_list ap;
                va_start(ap, msg);
                vsnprintf(msg_buf, BUFSIZE, msg, ap);
                va_end(ap);
                msg_buf[BUFSIZE] = '\0';
                fprintf(stderr, "%s - error %d (%s)\n", msg_buf, err, cudaGetErrorString(err));
                exit(1);
        }
}


static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
{
        if(err!=cudaSuccess)
        {
                fprintf(stderr,"%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n",msg,file_name,line_number,cudaGetErrorString(err));
                std::cin.get();
                exit(EXIT_FAILURE);
        }
}

#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)




texture<uchar4, 2, cudaReadModeNormalizedFloat> tex;


__global__
void imgcopy(uchar4 * input,uchar4 *output, int width, int height, int output_pitch_el)
{
        int row = blockDim.y*blockIdx.y + threadIdx.y;
        int col = blockDim.x*blockIdx.x + threadIdx.x;

        if (row < height && col < width) {
                float4 px = tex2D(tex, col, row);
                output[row*output_pitch_el+col] =
                        make_uchar4(px.x*255, px.y*255, px.z*255, px.w*255);
        }
}


__global__
void imgCopy(unsigned char* input, 
                                        unsigned char* output, 
                                        int width,
                                        int height,
                                        int colorWidthStep){
	//2D Index of current thread
        const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
        const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

        //Only valid threads perform memory I/O
        if((xIndex<width) && (yIndex<height))
        {
                //Location of colored pixel in input
                const int color_tid = yIndex * colorWidthStep + (3 * xIndex);
                

                output[color_tid]           = input[color_tid];
                output[color_tid+1]           = input[color_tid + 1];
                output[color_tid+2]           = input[color_tid + 2];

        }
}


void imageCopy(const cv::Mat& input, cv::Mat& output)
{
        //Calculate total number of bytes of input and output image
        const int colorBytes = input.step * input.rows;

        unsigned char *d_input, *d_output;

        //Allocate device memory
        SAFE_CALL(cudaMalloc<unsigned char>(&d_input,colorBytes),"CUDA Malloc Failed");
        SAFE_CALL(cudaMalloc<unsigned char>(&d_output,colorBytes),"CUDA Malloc Failed");

        //Copy data from OpenCV input image to device memory
        SAFE_CALL(cudaMemcpy(d_input,input.ptr(),colorBytes,cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");

        //Specify a reasonable block size
        const dim3 block(16,16);

        //Calculate grid size to cover the whole image
        const dim3 grid((input.cols + block.x - 1)/block.x, (input.rows + block.y - 1)/block.y);

        //Launch the color conversion kernel
        imgCopy<<<grid,block>>>(d_input,d_output,input.cols,input.rows,input.step);

        //Synchronize to check for any kernel launch errors
        SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");

        //Copy back data from destination device meory to OpenCV output image
        SAFE_CALL(cudaMemcpy(output.ptr(),d_output,colorBytes,cudaMemcpyDeviceToHost),"CUDA Memcpy Host To Device Failed");

        //Free the device memory
        SAFE_CALL(cudaFree(d_input),"CUDA Free Failed");
        SAFE_CALL(cudaFree(d_output),"CUDA Free Failed");
}




int main(int argc, char ** argv){
	std::string imagePath = "../img/gallo.png";

        //Read input image from the disk
        cv::Mat input = cv::imread(imagePath,cv::IMREAD_COLOR);

        if(input.empty())
        {
                std::cerr<<"Image Not Found!"<<std::endl;
                return -1;
        }

        //Create output image
        //std::cout<<cv::CV_8UC1<<std::endl;
        cv::Mat output(input.rows,input.cols,input.type());
        if (output.empty())
        {
                cout << "\n Image not created. You"
                     " have done something wrong. \n";
                return -1;    // Unsuccessful.
        }


        //Call the wrapper function
        imageCopy(input,output);

        cv::imwrite("starry_night.png",output);


}
