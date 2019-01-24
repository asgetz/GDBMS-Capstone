/////////////////////////

#ifndef AUX_FUNCTIONS_H
#define AUX_FUNCTIONS_H

/**Function to scale input on interval [0,1]
 *
 * params: int i, int n
 * */
float scale(int i, int n);

/**Compute the distance between 2 points on a line.
 *
 * params: float x1, float x2
 * */
float distance(float x1, float x2);

/**Compute scaled distance for an array of input values.
 *
 * params: float out, float in, float ref, int n
 * */
void distanceArray(float *out, float *in, float ref, int n);

#endif
