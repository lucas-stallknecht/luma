#ifndef RANDOM_GLSL_INCLUDED
#define RANDOM_GLSL_INCLUDED

#ifndef PI
#define PI 3.14159265359
#endif

uint pcg_hash(uint seed)
{
    uint state = seed * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float random_float(inout uint seed)
{
    seed = pcg_hash(seed);
    return float(seed) * (1.0 / 4294967295.0);
}

uint hash_uint3(uvec3 v)
{
    uint seed = v.x;
    seed = pcg_hash(seed ^ v.y);
    seed = pcg_hash(seed ^ v.z);
    return seed;
}

vec3 sample_cosine_weighted_hemisphere(uint seed) {
    float f1 = random_float(seed);
    float f2 = random_float(seed);

    float phi = 2.0 * PI * f1;
    float r = sqrt(f2);

    float x = r * cos(phi);
    float y = r * sin(phi);
    float z = sqrt(1.0 - f2);

    return normalize(vec3(x, y, z));
}

#endif
