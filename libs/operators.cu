
__host__ __device__ float4 operator+(float4 a, float4 b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
__host__ __device__ float4 operator/(float4 a, float4 b) { return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w); }
__host__ __device__ float4 operator/(float4 a, float b)  { return make_float4(a.x / b, a.y / b, a.z / b, a.w / b); }
__host__ __device__ float4 operator*(float4 a, float b)  { return make_float4(a.x * b, a.y * b, a.z * b, a.w * b); }
__host__ __device__ int2   operator+(int2 a, int2 b) { return make_int2(a.x + b.x, a.y + b.y); }

//Traduciamo dalle coordinate (x,y) a coordinate lineari
__host__ __device__ int tolinear(int2 coords, int width){
    return (coords.y * width) + coords.x;
}