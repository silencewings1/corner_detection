#pragma once
#include <algorithm>
// #include <cmath>
// #include <limits>

//////////////////////////////////////////////////////////////////
// math functions needed for rotation conversion.

// dot and cross production

template <typename T>
inline T DotProduct(const T x[3], const T y[3])
{
    return (x[0] * y[0] + x[1] * y[1] + x[2] * y[2]);
}

template <typename T>
inline void CrossProduct(const T x[3], const T y[3], T result[3])
{
    result[0] = x[1] * y[2] - x[2] * y[1];
    result[1] = x[2] * y[0] - x[0] * y[2];
    result[2] = x[0] * y[1] - x[1] * y[0];
}

template <typename T>
inline void RotPRY(const T* pose, T mat[9])
{
    const T phi = pose[0];
    const T theta = pose[1];
    const T psi = pose[2];

    mat[0] = cos(phi) * cos(theta);
    mat[1] = -sin(phi) * cos(psi) + cos(phi) * sin(theta) * sin(psi);
    mat[2] = sin(phi) * sin(psi) + cos(phi) * sin(theta) * cos(psi);
    mat[3] = sin(phi) * cos(theta);
    mat[4] = cos(phi) * cos(psi) + sin(phi) * sin(theta) * sin(psi);
    mat[5] = -cos(phi) * sin(psi) + sin(phi) * sin(theta) * cos(psi);
    mat[6] = -sin(theta);
    mat[7] = cos(theta) * sin(psi);
    mat[8] = cos(theta) * cos(psi);
}

template <typename T>
inline void Pose2RT(const T* pose, T R[9], T t[3])
{
    RotPRY(pose, R);
    t[0] = pose[3];
    t[1] = pose[4];
    t[2] = pose[5];
}

template <typename T>
inline void MatMulMat(const T A[9], const T B[9], T result[9])
{
    T A_row_1[3] = {A[0], A[1], A[2]};
    T A_row_2[3] = {A[3], A[4], A[5]};
    T A_row_3[3] = {A[6], A[7], A[8]};

    T B_col_1[3] = {B[0], B[3], B[6]};
    T B_col_2[3] = {B[1], B[4], B[7]};
    T B_col_3[3] = {B[2], B[5], B[8]};

    result[0] = DotProduct(A_row_1, B_col_1);
    result[1] = DotProduct(A_row_1, B_col_2);
    result[2] = DotProduct(A_row_1, B_col_3);

    result[3] = DotProduct(A_row_2, B_col_1);
    result[4] = DotProduct(A_row_2, B_col_2);
    result[5] = DotProduct(A_row_2, B_col_3);

    result[6] = DotProduct(A_row_3, B_col_1);
    result[7] = DotProduct(A_row_3, B_col_2);
    result[8] = DotProduct(A_row_3, B_col_3);
}

template <typename T>
inline void MatMulVec(const T A[9], const T B[3], T result[3])
{
    T A_row_1[3] = {A[0], A[1], A[2]};
    T A_row_2[3] = {A[3], A[4], A[5]};
    T A_row_3[3] = {A[6], A[7], A[8]};

    result[0] = DotProduct(A_row_1, B);
    result[1] = DotProduct(A_row_2, B);
    result[2] = DotProduct(A_row_3, B);
}

template <typename T>
inline void VecAddVec(const T A[3], const T B[3], T result[3])
{
    result[0] = A[0] + B[0];
    result[1] = A[1] + B[1];
    result[2] = A[2] + B[2];
}