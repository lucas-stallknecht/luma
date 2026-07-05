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

vec3 sample_cosine_weighted_hemisphere(inout uint seed) {
    float f1 = random_float(seed);
    float f2 = random_float(seed);

    float phi = 2.0 * PI * f1;
    float r = sqrt(f2);

    float x = r * cos(phi);
    float y = r * sin(phi);
    float z = sqrt(1.0 - f2);

    return normalize(vec3(x, y, z));
}

vec3 sample_uniform_sphere(inout uint seed) {
    float f1 = random_float(seed);
    float f2 = random_float(seed);

    float z = 1.0 - 2.0 * f1;
    float r = sqrt(max(0.0, 1.0 - z * z));
    float phi = 2.0 * PI * f2;

    float x = r * cos(phi);
    float y = r * sin(phi);

    return vec3(x, y, z);
}

vec3 fibonacci_sphere(uint i, uint n) {
    const float GOLDEN_ANGLE = 2.39996322972865332; // pi * (3.0 - sqrt(5.0))

    float z = 1.0 - (2.0 * float(i) + 1.0) / float(n);
    float r = sqrt(max(0.0, 1.0 - z * z));
    float phi = float(i) * GOLDEN_ANGLE;

    return vec3(r * cos(phi), r * sin(phi), z);
}

// uniformly random rotation matrix (Shoemake's quaternion method). Rotating the
// Fibonacci sphere by a fresh one each bake decorrelates neighbouring probes and
// lets accumulation over frames keep refining instead of resampling the same set
mat3 random_rotation(inout uint seed) {
    float u1 = random_float(seed);
    float u2 = random_float(seed);
    float u3 = random_float(seed);

    float sq1 = sqrt(1.0 - u1);
    float sq2 = sqrt(u1);
    float t1 = 2.0 * PI * u2;
    float t2 = 2.0 * PI * u3;

    // random unit quaternion (x, y, z, w)
    float x = sq1 * sin(t1);
    float y = sq1 * cos(t1);
    float z = sq2 * sin(t2);
    float w = sq2 * cos(t2);

    return mat3(
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w), 2.0 * (x * z - y * w),
        2.0 * (x * y - z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + x * w),
        2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x * x + y * y)
    );
}

#endif
