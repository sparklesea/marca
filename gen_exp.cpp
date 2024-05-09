#include <iostream>
#include <cmath>
#include <fstream>

float fast_exp(float y) {
    union
    {
        uint32_t i;
        float f;
    }v;
    
    v.i=((1<<23)*(1.4426950409*y+126.94201519f));
    // return v.f + 0.0285784f;
    return v.f;
}

float fast_log(float y){
    union
    {
        uint32_t i;
        float f;
    }v;
    v.f = y;
    return (((float)v.i)/(1<<23)-126.94201519f)/1.4426950409;
    // return (((float)v.i)/(1<<23)-127.0f)/1.4426950409;
}

float fast_softplus(float y){
    union{
        uint32_t i;
        float f;
    }v;
    v.i=((1<<23)*(1.4426950409*y+126.94201519f));
    // v.f+=(1-0.0523411);
    v.f+=1;
    return (((float)v.i)/(1<<23)-126.94201519f)/1.4426950409;
}

// int main() {

//     float min_value = -0.4;
//     float max_value = 0.0;
//     int num_points = 100;

//     float diff = 0;
//     // 生成 exponential 函数
//     for (int i = 1; i < num_points+1; ++i) {
//         float x = min_value + 0.4/num_points*i;
//         float result = exp(x);
//         float ours = fast_exp(x);
//         // std::cout<<std::abs(result-ours)<<std::endl;
//         diff += std::abs(result - ours);
//         // std::cout  << "x: " << x << ", ref: " << result << ", ours: " << ours << ", diff: " << ours-result << std::endl;
//         std::cout  << ours << std::endl;
//     }
//     // std::cout<<diff/num_points<<std::endl;
//     return 0;
// }

int main() {

    float min_value = -4;
    float max_value = -2;
    int num_points = 50;

    float diff = 0;
    // 生成 exponential 函数
    for (int i = 1; i < num_points+1; ++i) {
        float x = min_value + (max_value-min_value)/num_points*i;
        float result = std::log(1+std::exp(x));
        float ours = fast_log(1-0.0423411+fast_exp(x));
        // float ours = fast_log(1-0.0235627+fast_exp(x));
        // float ours = fast_softplus(x);
        // std::cout<<std::abs(result-ours)<<std::endl;
        diff += std::abs(result - ours);
        std::cout  << "x: " << x << ", ref: " << result << ", ours: " << ours << ", diff: " << ours-result << std::endl;
        // std::cout  << ours << std::endl;
    }
    std::cout<<diff/num_points<<std::endl;
    return 0;
}