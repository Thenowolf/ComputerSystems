// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Paralel Systems
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage
// Multiplication of elements in float array
//
// ***********************************************************************

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <stdio.h>

#include <string.h>
#include <stdint.h>

__device__ const uint32_t k[64] = {
0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee ,
0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501 ,
0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be ,
0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821 ,
0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa ,
0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8 ,
0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed ,
0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a ,
0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c ,
0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70 ,
0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05 ,
0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665 ,
0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039 ,
0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1 ,
0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1 ,
0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391 };
// r specifies the per-round shift amounts
__device__ const uint32_t r[] = {7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
                      5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20,
                      4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
                      6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21};
// leftrotate function definition
#define LEFTROTATE(x, c) (((x) << (c)) | ((x) >> (32 - (c))))
__device__ void to_bytes(uint32_t val, uint8_t *bytes)
{
    bytes[0] = (uint8_t) val;
    bytes[1] = (uint8_t) (val >> 8);
    bytes[2] = (uint8_t) (val >> 16);
    bytes[3] = (uint8_t) (val >> 24);
}
__device__ uint32_t to_int32(const uint8_t *bytes)
{
    return (uint32_t) bytes[0]
        | ((uint32_t) bytes[1] << 8)
        | ((uint32_t) bytes[2] << 16)
        | ((uint32_t) bytes[3] << 24);
}
__device__ void md5(uint8_t *initial_msg, size_t initial_len, uint8_t *digest) {
    // These vars will contain the hash
    uint32_t h0, h1, h2, h3;
    // Message (to prepare)
    uint8_t *msg = NULL;
    size_t new_len, offset;
    uint32_t w[16];
    uint32_t a, b, c, d, i, f, g, temp;
    // Initialize variables - simple count in nibbles:
    h0 = 0x67452301;
    h1 = 0xefcdab89;
    h2 = 0x98badcfe;
    h3 = 0x10325476;
    //Pre-processing:
    //append "1" bit to message
    //append "0" bits until message length in bits ≡ 448 (mod 512)
    //append length mod (2^64) to message
    for (new_len = initial_len + 1; new_len % (512/8) != 448/8; new_len++)
        ;
    msg = (uint8_t*)malloc(new_len + 8);
    memcpy(msg, initial_msg, initial_len);
    msg[initial_len] = 0x80; // append the "1" bit; most significant bit is "first"
    for (offset = initial_len + 1; offset < new_len; offset++)
        msg[offset] = 0; // append "0" bits
    // append the len in bits at the end of the buffer.
    to_bytes(initial_len*8, msg + new_len);
    // initial_len>>29 == initial_len*8>>32, but avoids overflow.
    to_bytes(initial_len>>29, msg + new_len + 4);
    // Process the message in successive 512-bit chunks:
    //for each 512-bit chunk of message:
    for(offset=0; offset<new_len; offset += (512/8)) {
        // break chunk into sixteen 32-bit words w[j], 0 ≤ j ≤ 15
        for (i = 0; i < 16; i++)
            w[i] = to_int32(msg + offset + i*4);
        // Initialize hash value for this chunk:
        a = h0;
        b = h1;
        c = h2;
        d = h3;
        // Main loop:
        for(i = 0; i<64; i++) {
            if (i < 16) {
                f = (b & c) | ((~b) & d);
                g = i;
            } else if (i < 32) {
                f = (d & b) | ((~d) & c);
                g = (5*i + 1) % 16;
            } else if (i < 48) {
                f = b ^ c ^ d;
                g = (3*i + 5) % 16;
            } else {
                f = c ^ (b | (~d));
                g = (7*i) % 16;
            }
            temp = d;
            d = c;
            c = b;
            b = b + LEFTROTATE((a + f + k[i] + w[g]), r[i]);
            a = temp;
        }
        // Add this chunk's hash to result so far:
        h0 += a;
        h1 += b;
        h2 += c;
        h3 += d;
    }
    // cleanup
    cudaFree(msg);
    //var char digest[16] := h0 append h1 append h2 append h3 //(Output is in little-endian)
    to_bytes(h0, digest);
    to_bytes(h1, digest + 4);
    to_bytes(h2, digest + 8);
    to_bytes(h3, digest + 12);
}

/* F, G and H are basic MD5 functions: selection, majority, parity */
#define F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define G(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define H(x, y, z) ((x) ^ (y) ^ (z))
#define I(x, y, z) ((y) ^ ((x) | (~z)))

/* ROTATE_LEFT rotates x left n bits */
#define ROTATE_LEFT(x, n) (((x) << (n)) | ((x) >> (32-(n))))

/* FF, GG, HH, and II transformations for rounds 1, 2, 3, and 4 */
/* Rotation is separate from addition to prevent recomputation */
#define FF(a, b, c, d, x, s, ac) \
  {(a) += F ((b), (c), (d)) + (x) + (uint)(ac); \
   (a) = ROTATE_LEFT ((a), (s)); \
   (a) += (b); \
  }
#define GG(a, b, c, d, x, s, ac) \
  {(a) += G ((b), (c), (d)) + (x) + (uint)(ac); \
   (a) = ROTATE_LEFT ((a), (s)); \
   (a) += (b); \
  }
#define HH(a, b, c, d, x, s, ac) \
  {(a) += H ((b), (c), (d)) + (x) + (uint)(ac); \
   (a) = ROTATE_LEFT ((a), (s)); \
   (a) += (b); \
  }
#define II(a, b, c, d, x, s, ac) \
  {(a) += I ((b), (c), (d)) + (x) + (uint)(ac); \
   (a) = ROTATE_LEFT ((a), (s)); \
   (a) += (b); \
  }

// Demo kernel for array elements multiplication.
// Every thread selects one element and multiply it.

__device__ void md5_vfy(char* data, uint length, uint8_t *digest)
{

	const uint a0 = 0x67452301;
	const uint b0 = 0xEFCDAB89;
	const uint c0 = 0x98BADCFE;
	const uint d0 = 0x10325476;

	uint a = 0;
    uint b = 0;
    uint c = 0;
    uint d = 0;

uint vals[14] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0};

int i = 0;

for(i=0; i < length; i++)
{
    vals[i / 4] |= data[i] << ((i % 4) * 8);
}
vals[i / 4] |= 0x80 << ((i % 4) * 8);

uint bitlen = length * 8;

#define in0  (vals[0])//x
#define in1  (vals[1])//y
#define in2  (vals[2])//z
#define in3  (vals[3])
#define in4  (vals[4])
#define in5  (vals[5])
#define in6  (vals[6])
#define in7  (vals[7])
#define in8  (vals[8])
#define in9  (vals[9])
#define in10 (vals[10])
#define in11 (vals[11])
#define in12 (vals[12])
#define in13 (vals[13])
#define in14 (bitlen) //w = bit length
#define in15 (0)

	//Initialize hash value for this chunk:
	a = a0;
	b = b0;
	c = c0;
	d = d0;


  /* Round 1 */
#define S11 7
#define S12 12
#define S13 17
#define S14 22
  FF ( a, b, c, d, in0,  S11, 3614090360); /* 1 */
  FF ( d, a, b, c, in1,  S12, 3905402710); /* 2 */
  FF ( c, d, a, b, in2,  S13,  606105819); /* 3 */
  FF ( b, c, d, a, in3,  S14, 3250441966); /* 4 */
  FF ( a, b, c, d, in4,  S11, 4118548399); /* 5 */
  FF ( d, a, b, c, in5,  S12, 1200080426); /* 6 */
  FF ( c, d, a, b, in6,  S13, 2821735955); /* 7 */
  FF ( b, c, d, a, in7,  S14, 4249261313); /* 8 */
  FF ( a, b, c, d, in8,  S11, 1770035416); /* 9 */
  FF ( d, a, b, c, in9,  S12, 2336552879); /* 10 */
  FF ( c, d, a, b, in10, S13, 4294925233); /* 11 */
  FF ( b, c, d, a, in11, S14, 2304563134); /* 12 */
  FF ( a, b, c, d, in12, S11, 1804603682); /* 13 */
  FF ( d, a, b, c, in13, S12, 4254626195); /* 14 */
  FF ( c, d, a, b, in14, S13, 2792965006); /* 15 */
  FF ( b, c, d, a, in15, S14, 1236535329); /* 16 */

  /* Round 2 */
#define S21 5
#define S22 9
#define S23 14
#define S24 20
  GG ( a, b, c, d, in1, S21, 4129170786); /* 17 */
  GG ( d, a, b, c, in6, S22, 3225465664); /* 18 */
  GG ( c, d, a, b, in11, S23,  643717713); /* 19 */
  GG ( b, c, d, a, in0, S24, 3921069994); /* 20 */
  GG ( a, b, c, d, in5, S21, 3593408605); /* 21 */
  GG ( d, a, b, c, in10, S22,   38016083); /* 22 */
  GG ( c, d, a, b, in15, S23, 3634488961); /* 23 */
  GG ( b, c, d, a, in4, S24, 3889429448); /* 24 */
  GG ( a, b, c, d, in9, S21,  568446438); /* 25 */
  GG ( d, a, b, c, in14, S22, 3275163606); /* 26 */
  GG ( c, d, a, b, in3, S23, 4107603335); /* 27 */
  GG ( b, c, d, a, in8, S24, 1163531501); /* 28 */
  GG ( a, b, c, d, in13, S21, 2850285829); /* 29 */
  GG ( d, a, b, c, in2, S22, 4243563512); /* 30 */
  GG ( c, d, a, b, in7, S23, 1735328473); /* 31 */
  GG ( b, c, d, a, in12, S24, 2368359562); /* 32 */

  /* Round 3 */
#define S31 4
#define S32 11
#define S33 16
#define S34 23
  HH ( a, b, c, d, in5, S31, 4294588738); /* 33 */
  HH ( d, a, b, c, in8, S32, 2272392833); /* 34 */
  HH ( c, d, a, b, in11, S33, 1839030562); /* 35 */
  HH ( b, c, d, a, in14, S34, 4259657740); /* 36 */
  HH ( a, b, c, d, in1, S31, 2763975236); /* 37 */
  HH ( d, a, b, c, in4, S32, 1272893353); /* 38 */
  HH ( c, d, a, b, in7, S33, 4139469664); /* 39 */
  HH ( b, c, d, a, in10, S34, 3200236656); /* 40 */
  HH ( a, b, c, d, in13, S31,  681279174); /* 41 */
  HH ( d, a, b, c, in0, S32, 3936430074); /* 42 */
  HH ( c, d, a, b, in3, S33, 3572445317); /* 43 */
  HH ( b, c, d, a, in6, S34,   76029189); /* 44 */
  HH ( a, b, c, d, in9, S31, 3654602809); /* 45 */
  HH ( d, a, b, c, in12, S32, 3873151461); /* 46 */
  HH ( c, d, a, b, in15, S33,  530742520); /* 47 */
  HH ( b, c, d, a, in2, S34, 3299628645); /* 48 */

  /* Round 4 */
#define S41 6
#define S42 10
#define S43 15
#define S44 21
  II ( a, b, c, d, in0, S41, 4096336452); /* 49 */
  II ( d, a, b, c, in7, S42, 1126891415); /* 50 */
  II ( c, d, a, b, in14, S43, 2878612391); /* 51 */
  II ( b, c, d, a, in5, S44, 4237533241); /* 52 */
  II ( a, b, c, d, in12, S41, 1700485571); /* 53 */
  II ( d, a, b, c, in3, S42, 2399980690); /* 54 */
  II ( c, d, a, b, in10, S43, 4293915773); /* 55 */
  II ( b, c, d, a, in1, S44, 2240044497); /* 56 */
  II ( a, b, c, d, in8, S41, 1873313359); /* 57 */
  II ( d, a, b, c, in15, S42, 4264355552); /* 58 */
  II ( c, d, a, b, in6, S43, 2734768916); /* 59 */
  II ( b, c, d, a, in13, S44, 1309151649); /* 60 */
  II ( a, b, c, d, in4, S41, 4149444226); /* 61 */
  II ( d, a, b, c, in11, S42, 3174756917); /* 62 */
  II ( c, d, a, b, in2, S43,  718787259); /* 63 */
  II ( b, c, d, a, in9, S44, 3951481745); /* 64 */

	a += a0;
	b += b0;
	c += c0;
	d += d0;

    //*a1 = a;
    //*b1 = b;
    //*c1 = c;
    //*d1 = d;
    to_bytes(a, digest);
    to_bytes(b, digest + 4);
    to_bytes(c, digest + 8);
    to_bytes(d, digest + 12);
}

__global__ void kernel_mult(char *pole, unsigned long long int  L, uint8_t * result, unsigned long long int strlen )
{
	unsigned long long int l = (unsigned long long int )blockDim.x * (unsigned long long int )blockIdx.x + (unsigned long long int )threadIdx.x;

    /*if (l == 99999){
        printf("%lu\n", (unsigned long long int )threadIdx.x);
        printf("%lu\n", (unsigned long long int )blockIdx.x);
        printf("%lu\n", (unsigned long long int )blockDim.x);
	} */
	//printf("L je %d\n",L);
	//printf("%d\n",l);
	//printf("%d\n",strlen);
	unsigned long long int stringStart = l * strlen;
	unsigned long long int resultStart = l * 16;
	unsigned long long int abecedaLen = 26;
    uint8_t result2[16];

    //unsigned long long int u = l;
	//char Rarray[3];
	//uint8_t result[16];
	// if grid is greater then length of array...
	//printf("%s\n", "zkousimCUDA");
	//printf("%d\n", l);
	if ( l >= L ) return;
    //printf("L je : %llu/n",L);
    /*if (l == 11881376){
        printf("dosel jsem tu");
    }*/

    /*if (l == 0){
        printf("dosel jsem tu\n");
        printf("stringstart: %llu\n",stringStart);
        printf("resultstart: %llu\n",resultStart);
    }*/
    char *z = new char[strlen];
    unsigned long long int help = l;
    /*if (u == 90000){
        printf("%llu", help);
	}*/
    for(int k = strlen-1; k >=0;k--){
			//printf("%d \n", k);
        z[k] = (help%abecedaLen) + 'a';
            //printf("%c\n",pole[k]);
        help = help / abecedaLen; // zajistím aby v další iteraci byl jiný char
            if (l == 32){
               // printf("%c\n",pole[k]);
               // printf("%d\n",k);
                //printf("%llu\n",(l * strlen) + strlen);
            }
                //printf("%llu\n",(l * strlen) + strlen);
        }
                //printf("%s \n", vel);
		          //printf("\n");
                  if (l == 31){
                    printf("dosel jsem tu2\n");
    }

	/*for(unsigned long long int i = (l * strlen); i < ((l * strlen) + strlen); i++)
    { */
    	/*unsigned long long int help = l;
        for(unsigned long long int k = ((strlen-1) + l * strlen); k >=(0+l*strlen);k--){
			//printf("%d \n", k);
            pole[k] = (help%abecedaLen) + 'a';
            //printf("%c\n",pole[k]);
            help = help / abecedaLen; // zajistím aby v další iteraci byl jiný char
            if (l == 1){
                printf("%c\n",pole[k]);
                printf("%llu\n",k);
                printf("%llu\n",(l * strlen) + strlen);
            }
                //printf("%llu\n",(l * strlen) + strlen);
        }
                //printf("%s \n", vel);
		          //printf("\n");
                  if (l == 31){
                    printf("dosel jsem tu2\n");
    } */
    //}


	//printf("%s\n", "zkousimCUDA");
	md5_vfy(z,strlen,result2);
	//printf("%llu\n", l);
	/*for (int i = stringStart; i < stringStart+1; i++){
		printf("%d\n", i);
		//Rarray[y] = pole[i];
		y++;
	}*/
	//int y = 0;
	//printf("%d\n", y);
    
	//md5((uint8_t*)(z),strlen,result2);
    free(z);

    /*if (l == 11881375){
        printf("%lu\n", (unsigned long long int )threadIdx.x);
        printf("%lu\n", (unsigned long long int )blockIdx.x);
        printf("%lu\n", (unsigned long long int )blockDim.x);
	} */

    if (l == L-1){
        for (int i = 0; i < 16; i++){
        printf("%2.2x", result2[i]);
	}
    }

    free(result2); 

    //cudaFree( result2 );
    //cudaFree( z );
	//printf("%d\n", y);
	/*for (int i = 0; i < 16; i++)
        printf("%2.2x", result[i]);
    printf("\n"); */
	/*printf("%s\n", "zkousimCUDA");
	md5((uint8_t*)Rarray,3,result);

	for (int i = 0; i < 16; i++)
        printf("%2.2x", result[i]);
    printf("\n"); */
	//printf( "\n" );
	//printf("%c\n", pole[l]);
	//pole[i] = 'b';
	//printf("%c\n", pole[i]);
}

void run_mult(char *cudaP, unsigned long long int Length, uint8_t * result, unsigned long long int strlen  )
{
	cudaError_t cerr;
	unsigned long long int threads = 128;
	unsigned long long int blocks = ( Length + threads - 1 ) / threads;

	//char *cudaP;

	// Memory allocation in GPU device
	//unsigned char *cudaP[200];

	/*cerr = cudaMalloc( &cudaP, Length * strlen /*sizeof( char )  );
	if ( cerr != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cerr ) );	 */

	// Copy data from PC to GPU device
	/*cerr = cudaMemcpy( cudaP, P, Length * sizeof( char ), cudaMemcpyHostToDevice );
	if ( cerr != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cerr ) ); */	
	
	// Grid creation
	kernel_mult<<< blocks, threads >>>( cudaP, Length, result, strlen );
	//
	if ( ( cerr = cudaGetLastError() ) != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cerr ) );
	
	// Copy data from GPU device to PC
	/*cerr = cudaMemcpy( P, cudaP, Length * strlen /*sizeof( char ), cudaMemcpyDeviceToHost );
	if ( cerr != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cerr ) ); */
	
	// Free memory
	//cudaFree( cudaP );
	cudaDeviceSynchronize();
}