#ifndef RAY_UTILS_GLSL_INCLUDED
#define RAY_UTILS_GLSL_INCLUDED

#extension GL_EXT_ray_query : require

// Duff et al., "Building an Orthonormal Basis, Revisited"
mat3 build_onb(vec3 n) {
    float sign = n.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (sign + n.z);
    float b = n.x * n.y * a;
    vec3 t = vec3(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
    vec3 bt = vec3(b, sign + n.y * n.y * a, -n.y);
    return mat3(t, bt, n);
}

bool trace_occluded(accelerationStructureEXT tlas, vec3 origin, vec3 dir, float t_max) {
    rayQueryEXT ray_query;
    rayQueryInitializeEXT(
        ray_query, tlas, gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT,
        0xFFu, origin, 0.001, dir, t_max
    );
    rayQueryProceedEXT(ray_query);
    return rayQueryGetIntersectionTypeEXT(ray_query, true) != gl_RayQueryCommittedIntersectionNoneEXT;
}

#endif
