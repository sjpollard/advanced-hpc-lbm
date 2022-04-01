/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(ii)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(jj) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   ./d2q9-bgk input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <malloc.h>
#include <mpi.h>

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"

/* struct to hold the parameter values */
typedef struct
{
  int    global_nx;            /* no. of cells in x-direction */
  int    global_ny;            /* no. of cells in y-direction */
  int    nx;
  int    ny;
  int    start_row;
  int    end_row;
  int    global_cells;   
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
} t_param;

/* struct to hold the 'speed' values */
typedef struct
{
  //float speeds[NSPEEDS];
  float* restrict s0;
  float* restrict s1;
  float* restrict s2;
  float* restrict s3;
  float* restrict s4;
  float* restrict s5;
  float* restrict s6;
  float* restrict s7;
  float* restrict s8;
} t_speed;

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed* cells_ptr, t_speed* tmp_cells_ptr,
               t_speed* send_buffer, t_speed* receive_buffer,
               int** obstacles_ptr, float** av_vels_ptr,
               float** av_vels_buffer_ptr, int rank, int size);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
float timestep(const t_param params, float* restrict cells_s0, float* restrict cells_s1, float* restrict cells_s2, 
                                   float* restrict cells_s3, float* restrict cells_s4, float* restrict cells_s5, 
                                   float* restrict cells_s6, float* restrict cells_s7, float* restrict cells_s8, 
                                   float* restrict tmp_cells_s0, float* restrict tmp_cells_s1, float* restrict tmp_cells_s2, 
                                   float* restrict tmp_cells_s3, float* restrict tmp_cells_s4, float* restrict tmp_cells_s5, 
                                   float* restrict tmp_cells_s6, float* restrict tmp_cells_s7, float* restrict tmp_cells_s8, int* obstacles);
int accelerate_flow(const t_param params, float* restrict cells_s0, float* restrict cells_s1, float* restrict cells_s2, 
                                   float* restrict cells_s3, float* restrict cells_s4, float* restrict cells_s5, 
                                   float* restrict cells_s6, float* restrict cells_s7, float* restrict cells_s8, int* restrict obstacles);
float grid_ops(const t_param params, const float* restrict cells_s0, const float* restrict cells_s1, const float* restrict cells_s2, 
                                   const float* restrict cells_s3, const float* restrict cells_s4, const float* restrict cells_s5, 
                                   const float* restrict cells_s6, const float* restrict cells_s7, const float* restrict cells_s8, 
                                   float* restrict tmp_cells_s0, float* restrict tmp_cells_s1, float* restrict tmp_cells_s2, 
                                   float* restrict tmp_cells_s3, float* restrict tmp_cells_s4, float* restrict tmp_cells_s5, 
                                   float* restrict tmp_cells_s6, float* restrict tmp_cells_s7, float* restrict tmp_cells_s8, int* restrict obstacles);
int write_values(const t_param params, t_speed cells, int* obstacles, float* av_vels, int rank, int size);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed* cells_ptr, t_speed* tmp_cells_ptr,
             t_speed* send_buffer, t_speed* receive_buffer,
             int** obstacles_ptr, float** av_vels_ptr, float** av_vels_buffer_ptr, int rank);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed cells);

/* compute average velocity */
float av_velocity(const t_param params, float* restrict cells_s0, float* restrict cells_s1, float* restrict cells_s2, 
                                   float* restrict cells_s3, float* restrict cells_s4, float* restrict cells_s5, 
                                   float* restrict cells_s6, float* restrict cells_s7, float* restrict cells_s8, int* restrict obstacles);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed cells, int* obstacles, float av_vel);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);

  MPI_Status status;
  int rank;
  int size;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int above = (rank + 1) % size;
  int below = (rank == 0) ? (size - 1) : (rank - 1);

  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  t_speed cells;     //= NULL;    /* grid containing fluid densities */
  t_speed tmp_cells; //= NULL;    /* scratch space */
  t_speed send_buffer;
  t_speed receive_buffer;
  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  float* av_vels_buffer = NULL;
  struct timeval timstr;                                                             /* structure to hold elapsed time */
  double tot_tic, tot_toc, init_tic, init_toc, comp_tic, comp_toc, col_tic, col_toc; /* floating point numbers to calculate elapsed wallclock time */
  int tot_cells = 0;

  /* parse the command line */
  if (argc != 3)
  {
    usage(argv[0]);
  }
  else
  {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }
  /* Total/init time starts here: initialise our data structures and load values from file */
  gettimeofday(&timstr, NULL);
  tot_tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  init_tic=tot_tic;
  initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &send_buffer, &receive_buffer, &obstacles, &av_vels, &av_vels_buffer, rank, size);
  for (int i = 0; i < params.nx * params.ny; i++) {
    tot_cells += (!obstacles[i]) ? 1 : 0;
  }
  MPI_Allreduce(&tot_cells, &(params.global_cells), 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD);

  /* Init time stops here, compute time starts*/
  gettimeofday(&timstr, NULL);
  init_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  comp_tic=init_toc;

  for (int tt = 0; tt < params.maxIters; tt++)
  {
    //send up, receive down
    for (int ii = 0; ii < params.nx; ii++){
      send_buffer.s0[ii] = cells.s0[ii + params.ny * params.nx];
      send_buffer.s1[ii] = cells.s1[ii + params.ny * params.nx];
      send_buffer.s2[ii] = cells.s2[ii + params.ny * params.nx];
      send_buffer.s3[ii] = cells.s3[ii + params.ny * params.nx];
      send_buffer.s4[ii] = cells.s4[ii + params.ny * params.nx];
      send_buffer.s5[ii] = cells.s5[ii + params.ny * params.nx];
      send_buffer.s6[ii] = cells.s6[ii + params.ny * params.nx];
      send_buffer.s7[ii] = cells.s7[ii + params.ny * params.nx];
      send_buffer.s8[ii] = cells.s8[ii + params.ny * params.nx];
    }
    MPI_Sendrecv(send_buffer.s0, params.nx, MPI_FLOAT, above, 0,
                 receive_buffer.s0, params.nx, MPI_FLOAT, below, 0,
                 MPI_COMM_WORLD, &status);
    MPI_Sendrecv(send_buffer.s1, params.nx, MPI_FLOAT, above, 0,
                 receive_buffer.s1, params.nx, MPI_FLOAT, below, 0,
                 MPI_COMM_WORLD, &status);
    MPI_Sendrecv(send_buffer.s2, params.nx, MPI_FLOAT, above, 0,
                 receive_buffer.s2, params.nx, MPI_FLOAT, below, 0,
                 MPI_COMM_WORLD, &status);
    MPI_Sendrecv(send_buffer.s3, params.nx, MPI_FLOAT, above, 0,
                 receive_buffer.s3, params.nx, MPI_FLOAT, below, 0,
                 MPI_COMM_WORLD, &status);
    MPI_Sendrecv(send_buffer.s4, params.nx, MPI_FLOAT, above, 0,
                 receive_buffer.s4, params.nx, MPI_FLOAT, below, 0,
                 MPI_COMM_WORLD, &status);
    MPI_Sendrecv(send_buffer.s5, params.nx, MPI_FLOAT, above, 0,
                 receive_buffer.s5, params.nx, MPI_FLOAT, below, 0,
                 MPI_COMM_WORLD, &status);
    MPI_Sendrecv(send_buffer.s6, params.nx, MPI_FLOAT, above, 0,
                 receive_buffer.s6, params.nx, MPI_FLOAT, below, 0,
                 MPI_COMM_WORLD, &status);
    MPI_Sendrecv(send_buffer.s7, params.nx, MPI_FLOAT, above, 0,
                 receive_buffer.s7, params.nx, MPI_FLOAT, below, 0,
                 MPI_COMM_WORLD, &status);
    MPI_Sendrecv(send_buffer.s8, params.nx, MPI_FLOAT, above, 0,
                 receive_buffer.s8, params.nx, MPI_FLOAT, below, 0,
                 MPI_COMM_WORLD, &status);
    for (int ii = 0; ii < params.nx; ii++){
      cells.s0[ii] = receive_buffer.s0[ii];
      cells.s1[ii] = receive_buffer.s1[ii];
      cells.s2[ii] = receive_buffer.s2[ii];
      cells.s3[ii] = receive_buffer.s3[ii];
      cells.s4[ii] = receive_buffer.s4[ii];
      cells.s5[ii] = receive_buffer.s5[ii];
      cells.s6[ii] = receive_buffer.s6[ii];
      cells.s7[ii] = receive_buffer.s7[ii];
      cells.s8[ii] = receive_buffer.s8[ii];
    }

    //send down, receive up
    for (int ii = 0; ii < params.nx; ii++){
      send_buffer.s0[ii] = cells.s0[ii + params.nx];
      send_buffer.s1[ii] = cells.s1[ii + params.nx];
      send_buffer.s2[ii] = cells.s2[ii + params.nx];
      send_buffer.s3[ii] = cells.s3[ii + params.nx];
      send_buffer.s4[ii] = cells.s4[ii + params.nx];
      send_buffer.s5[ii] = cells.s5[ii + params.nx];
      send_buffer.s6[ii] = cells.s6[ii + params.nx];
      send_buffer.s7[ii] = cells.s7[ii + params.nx];
      send_buffer.s8[ii] = cells.s8[ii + params.nx];
    }
    MPI_Sendrecv(send_buffer.s0, params.nx, MPI_FLOAT, below, 0,
                 receive_buffer.s0, params.nx, MPI_FLOAT, above, 0,
                 MPI_COMM_WORLD, &status);
    MPI_Sendrecv(send_buffer.s1, params.nx, MPI_FLOAT, below, 0,
                 receive_buffer.s1, params.nx, MPI_FLOAT, above, 0,
                 MPI_COMM_WORLD, &status);
    MPI_Sendrecv(send_buffer.s2, params.nx, MPI_FLOAT, below, 0,
                 receive_buffer.s2, params.nx, MPI_FLOAT, above, 0,
                 MPI_COMM_WORLD, &status);
    MPI_Sendrecv(send_buffer.s3, params.nx, MPI_FLOAT, below, 0,
                 receive_buffer.s3, params.nx, MPI_FLOAT, above, 0,
                 MPI_COMM_WORLD, &status);
    MPI_Sendrecv(send_buffer.s4, params.nx, MPI_FLOAT, below, 0,
                 receive_buffer.s4, params.nx, MPI_FLOAT, above, 0,
                 MPI_COMM_WORLD, &status);
    MPI_Sendrecv(send_buffer.s5, params.nx, MPI_FLOAT, below, 0,
                 receive_buffer.s5, params.nx, MPI_FLOAT, above, 0,
                 MPI_COMM_WORLD, &status);
    MPI_Sendrecv(send_buffer.s6, params.nx, MPI_FLOAT, below, 0,
                 receive_buffer.s6, params.nx, MPI_FLOAT, above, 0,
                 MPI_COMM_WORLD, &status);
    MPI_Sendrecv(send_buffer.s7, params.nx, MPI_FLOAT, below, 0,
                 receive_buffer.s7, params.nx, MPI_FLOAT, above, 0,
                 MPI_COMM_WORLD, &status);
    MPI_Sendrecv(send_buffer.s8, params.nx, MPI_FLOAT, below, 0,
                 receive_buffer.s8, params.nx, MPI_FLOAT, above, 0,
                 MPI_COMM_WORLD, &status);
    for (int ii = 0; ii < params.nx; ii++){
      cells.s0[ii + (params.ny + 1) * params.nx] = receive_buffer.s0[ii];
      cells.s1[ii + (params.ny + 1) * params.nx] = receive_buffer.s1[ii];
      cells.s2[ii + (params.ny + 1) * params.nx] = receive_buffer.s2[ii];
      cells.s3[ii + (params.ny + 1) * params.nx] = receive_buffer.s3[ii];
      cells.s4[ii + (params.ny + 1) * params.nx] = receive_buffer.s4[ii];
      cells.s5[ii + (params.ny + 1) * params.nx] = receive_buffer.s5[ii];
      cells.s6[ii + (params.ny + 1) * params.nx] = receive_buffer.s6[ii];
      cells.s7[ii + (params.ny + 1) * params.nx] = receive_buffer.s7[ii];
      cells.s8[ii + (params.ny + 1) * params.nx] = receive_buffer.s8[ii];
    }

    av_vels[tt] = timestep(params, cells.s0, cells.s1, cells.s2, 
                     cells.s3, cells.s4, cells.s5, 
                     cells.s6, cells.s7, cells.s8, 
                     tmp_cells.s0, tmp_cells.s1, tmp_cells.s2, 
                     tmp_cells.s3, tmp_cells.s4, tmp_cells.s5, 
                     tmp_cells.s6, tmp_cells.s7, tmp_cells.s8, obstacles);
    //if (tt == 0) printf("Rank %d av_vel[0] %.12E  tot_u %.12E ny %d start_row %d end_row %d\n", rank, av_vels[tt], (av_vels[tt] * params.global_cells), params.ny, params.start_row, params.end_row);
    t_speed temp = cells;
    cells = tmp_cells;
    tmp_cells = temp;

#ifdef DEBUG
    printf("==timestep: %d==\n", tt);
    printf("av velocity: %.12E\n", av_vels[tt]);
    printf("tot density: %.12E\n", total_density(params, cells));
#endif
  }

  /* Compute time stops here, collate time starts*/
  gettimeofday(&timstr, NULL);
  comp_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  col_tic=comp_toc;

  // Collate data from ranks here 
  /*float av_vel = 0;
  for (int tt = 0; tt < params.maxIters; tt++) {
    MPI_Reduce(&(av_vels[tt]), &av_vel, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    av_vels[tt] = av_vel;
  }*/
  //printf("Rank %d rows %d av_vels[0] = %.12E\n", rank, params.ny, av_vels[0]);
  //printf("Rank: %d params.ny: %d start_row %d end_row %d\n", rank, params.ny, params.start_row, params. end_row);
  if (rank == 0 || rank == size - 1) printf("Rank: %d, above: %d, below: %d\n", rank, above, below);
  MPI_Reduce(av_vels, av_vels_buffer, params.maxIters, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
  //if (rank == 0) printf("Rank %d: av_vels_buffer[0] = %.12E\n", rank, av_vels_buffer[0]);

  /* Total/collate time stops here.*/
  gettimeofday(&timstr, NULL);
  col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  tot_toc = col_toc;

  /* write final values and free memory */
  if (rank == 0) {
    printf("==done==\n");
    printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles, av_vels_buffer[params.maxIters - 1]));
    printf("Elapsed Init time:\t\t\t%.6lf (s)\n",    init_toc - init_tic);
    printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
    printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc  - col_tic);
    printf("Elapsed Total time:\t\t\t%.6lf (s)\n",   tot_toc  - tot_tic);
  }
  write_values(params, cells, obstacles, av_vels_buffer, rank, size);
  finalise(&params, &cells, &tmp_cells, &send_buffer, &receive_buffer, &obstacles, &av_vels, &av_vels_buffer, rank);
  MPI_Finalize();

  return EXIT_SUCCESS;
}

float timestep(const t_param params, float* restrict cells_s0, float* restrict cells_s1, float* restrict cells_s2, 
                                   float* restrict cells_s3, float* restrict cells_s4, float* restrict cells_s5, 
                                   float* restrict cells_s6, float* restrict cells_s7, float* restrict cells_s8, 
                                   float* restrict tmp_cells_s0, float* restrict tmp_cells_s1, float* restrict tmp_cells_s2, 
                                   float* restrict tmp_cells_s3, float* restrict tmp_cells_s4, float* restrict tmp_cells_s5, 
                                   float* restrict tmp_cells_s6, float* restrict tmp_cells_s7, float* restrict tmp_cells_s8, int* obstacles)
{
  if (params.start_row <= params.global_ny - 2 && params.end_row >= params.global_ny - 2) {
    accelerate_flow(params, cells_s0, cells_s1, cells_s2, 
                     cells_s3, cells_s4, cells_s5, 
                     cells_s6, cells_s7, cells_s8, obstacles);
  }
  return grid_ops(params, cells_s0, cells_s1, cells_s2, 
                     cells_s3, cells_s4, cells_s5, 
                     cells_s6, cells_s7, cells_s8, tmp_cells_s0, tmp_cells_s1, tmp_cells_s2, 
                     tmp_cells_s3, tmp_cells_s4, tmp_cells_s5, 
                     tmp_cells_s6, tmp_cells_s7, tmp_cells_s8, obstacles);
}

int accelerate_flow(const t_param params, float* restrict cells_s0, float* restrict cells_s1, float* restrict cells_s2, 
                                   float* restrict cells_s3, float* restrict cells_s4, float* restrict cells_s5, 
                                   float* restrict cells_s6, float* restrict cells_s7, float* restrict cells_s8, int* restrict obstacles)
{
  /* compute weighting factors */
  const float w1 = params.density * params.accel / 9.f;
  const float w2 = params.density * params.accel / 36.f;

  /* modify the 2nd row of the grid */
  int jj = (params.ny == 1) ? 1 : params.ny - 1;
  __assume_aligned(obstacles, 64);
  __assume_aligned(cells_s0, 64);
  __assume_aligned(cells_s1, 64);
  __assume_aligned(cells_s2, 64);
  __assume_aligned(cells_s3, 64);
  __assume_aligned(cells_s4, 64);
  __assume_aligned(cells_s5, 64);
  __assume_aligned(cells_s6, 64);
  __assume_aligned(cells_s7, 64);
  __assume_aligned(cells_s8, 64);
  __assume((params.nx) % 2 == 0);
  __assume((params.nx) % 4 == 0);
  __assume((params.nx) % 8 == 0);
  __assume((params.nx) % 16 == 0);
  for (int ii = 0; ii < params.nx; ii++)
  {
    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[ii + (jj - 1)*params.nx]
        && (cells_s3[ii + jj*params.nx] - w1) > 0.f
        && (cells_s6[ii + jj*params.nx] - w2) > 0.f
        && (cells_s7[ii + jj*params.nx] - w2) > 0.f)
    {
      /* increase 'east-side' densities */
      cells_s1[ii + jj*params.nx] += w1;
      cells_s5[ii + jj*params.nx] += w2;
      cells_s8[ii + jj*params.nx] += w2;
      /* decrease 'west-side' densities */
      cells_s3[ii + jj*params.nx] -= w1;
      cells_s6[ii + jj*params.nx] -= w2;
      cells_s7[ii + jj*params.nx] -= w2;
    }
  }

  return EXIT_SUCCESS;
}

float grid_ops(const t_param params, const float* restrict cells_s0, const float* restrict cells_s1, const float* restrict cells_s2, 
                                   const float* restrict cells_s3, const float* restrict cells_s4, const float* restrict cells_s5, 
                                   const float* restrict cells_s6, const float* restrict cells_s7, const float* restrict cells_s8, 
                                   float* restrict tmp_cells_s0, float* restrict tmp_cells_s1, float* restrict tmp_cells_s2, 
                                   float* restrict tmp_cells_s3, float* restrict tmp_cells_s4, float* restrict tmp_cells_s5, 
                                   float* restrict tmp_cells_s6, float* restrict tmp_cells_s7, float* restrict tmp_cells_s8, int* restrict obstacles)
{ 
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */

  const float c_sq_i = 3.f;
  const float d1 = 0.5 * 3.f * 3.f;
  const float d2 = 0.5 * 3.f;

  float local_density_i;
  float tot_u = 0.00f;

  __assume_aligned(obstacles, 64);
  __assume_aligned(cells_s0, 64);
  __assume_aligned(cells_s1, 64);
  __assume_aligned(cells_s2, 64);
  __assume_aligned(cells_s3, 64);
  __assume_aligned(cells_s4, 64);
  __assume_aligned(cells_s5, 64);
  __assume_aligned(cells_s6, 64);
  __assume_aligned(cells_s7, 64);
  __assume_aligned(cells_s8, 64);
  __assume_aligned(tmp_cells_s0, 64);
  __assume_aligned(tmp_cells_s1, 64);
  __assume_aligned(tmp_cells_s2, 64);
  __assume_aligned(tmp_cells_s3, 64);
  __assume_aligned(tmp_cells_s4, 64);
  __assume_aligned(tmp_cells_s5, 64);
  __assume_aligned(tmp_cells_s6, 64);
  __assume_aligned(tmp_cells_s7, 64);
  __assume_aligned(tmp_cells_s8, 64);
  __assume((params.nx) % 2 == 0);
  __assume((params.nx) % 4 == 0);
  __assume((params.nx) % 8 == 0);
  __assume((params.nx) % 16 == 0);
  for (int jj = 1; jj < params.ny + 1; jj++)
  { 
    for (int ii = 0; ii < params.nx; ii++)
    {
      const int y_n = jj + 1;
      const int x_e = (ii + 1) % params.nx;
      const int y_s = jj - 1;
      const int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);
      /* propagate densities from neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */
      const float speed0 = cells_s0[ii + jj*params.nx]; /* central cell, no movement */
      const float speed1 = cells_s1[x_w + jj*params.nx]; /* east */
      const float speed2 = cells_s2[ii + y_s*params.nx]; /* north */
      const float speed3 = cells_s3[x_e + jj*params.nx]; /* west */
      const float speed4 = cells_s4[ii + y_n*params.nx]; /* south */
      const float speed5 = cells_s5[x_w + y_s*params.nx]; /* north-east */
      const float speed6 = cells_s6[x_e + y_s*params.nx]; /* north-west */
      const float speed7 = cells_s7[x_e + y_n*params.nx]; /* south-west */
      const float speed8 = cells_s8[x_w + y_n*params.nx]; /* south-east */

      //if (params.end_row == 127 && ii == 0 && jj == params.ny - 1) printf("jj: %d %.12E %.12E %.12E %.12E %.12E %.12E %.12E %.12E %.12E \n", jj, speed0, speed1, speed2, speed3, speed4, speed5, speed6, speed7, speed8);

      const float local_density = speed0 + speed1 + speed2+ speed3 + speed4 + speed5 + speed6 + speed7 + speed8;

      local_density_i = 1 / local_density;
      /* compute x velocity component */
      const float u_x = (speed1
                    + speed5
                    + speed8
                    - speed3
                       - speed6
                       - speed7)
                   * local_density_i;
      /* compute y velocity component */
      const float u_y = (speed2
                    + speed5
                    + speed6
                    - speed4
                       - speed7
                       - speed8)
                   * local_density_i;

      /* velocity squared */
      const float u_sq = u_x * u_x + u_y * u_y;

      /* directional velocity components */
      float u[NSPEEDS];
      u[1] =   u_x;        /* east */
      u[2] =         u_y;  /* north */
      u[3] = - u_x;        /* west */
      u[4] =       - u_y;  /* south */
      u[5] =   u_x + u_y;  /* north-east */
      u[6] = - u_x + u_y;  /* north-west */
      u[7] = - u_x - u_y;  /* south-west */
      u[8] =   u_x - u_y;  /* south-east */

      /* equilibrium densities */

      /* zero velocity density: weight w0 */
      const float d_equ0 = w0 * local_density
                 * (1.f - u_sq * d2);
      /* axis speeds: weight w1 */
      const float d_equ1 = w1 * local_density * (1.f + u[1] * c_sq_i
                                       + (u[1] * u[1]) * d1
                                       - u_sq * d2);
      const float d_equ2 = w1 * local_density * (1.f + u[2] * c_sq_i
                                       + (u[2] * u[2]) * d1
                                       - u_sq * d2);
      const float d_equ3 = w1 * local_density * (1.f + u[3] * c_sq_i
                                       + (u[3] * u[3]) * d1
                                       - u_sq * d2);
      const float d_equ4 = w1 * local_density * (1.f + u[4] * c_sq_i
                                       + (u[4] * u[4]) * d1
                                       - u_sq * d2);
      /* diagonal speeds: weight w2 */
      const float d_equ5 = w2 * local_density * (1.f + u[5] * c_sq_i
                                       + (u[5] * u[5]) * d1
                                       - u_sq * d2);
      const float d_equ6 = w2 * local_density * (1.f + u[6] * c_sq_i
                                       + (u[6] * u[6]) * d1
                                       - u_sq * d2);
      const float d_equ7 = w2 * local_density * (1.f + u[7] * c_sq_i
                                       + (u[7] * u[7]) * d1
                                       - u_sq * d2);
      const float d_equ8 = w2 * local_density * (1.f + u[8] * c_sq_i
                                       + (u[8] * u[8]) * d1
                                       - u_sq * d2);

      tmp_cells_s0[ii + jj*params.nx] = !obstacles[ii + (jj - 1)*params.nx] ? speed0
                                        + params.omega * (d_equ0 - speed0) : speed0;
      tmp_cells_s1[ii + jj*params.nx] = !obstacles[ii + (jj - 1)*params.nx] ? speed1
                                        + params.omega * (d_equ1 - speed1) : speed3;
      tmp_cells_s3[ii + jj*params.nx] = !obstacles[ii + (jj - 1)*params.nx] ? speed3
                                        + params.omega * (d_equ3 - speed3) : speed1;
      tmp_cells_s2[ii + jj*params.nx] = !obstacles[ii + (jj - 1)*params.nx] ? speed2
                                        + params.omega * (d_equ2 - speed2) : speed4;                                                                    
      tmp_cells_s4[ii + jj*params.nx] = !obstacles[ii + (jj - 1)*params.nx] ? speed4
                                        + params.omega * (d_equ4 - speed4) : speed2;
      tmp_cells_s5[ii + jj*params.nx] = !obstacles[ii + (jj - 1)*params.nx] ? speed5
                                        + params.omega * (d_equ5 - speed5) : speed7;
      tmp_cells_s7[ii + jj*params.nx] = !obstacles[ii + (jj - 1)*params.nx] ? speed7
                                        + params.omega * (d_equ7 - speed7) : speed5;
      tmp_cells_s6[ii + jj*params.nx] = !obstacles[ii + (jj - 1)*params.nx] ? speed6
                                        + params.omega * (d_equ6 - speed6) : speed8;
      tmp_cells_s8[ii + jj*params.nx] = !obstacles[ii + (jj - 1)*params.nx] ? speed8
                                        + params.omega * (d_equ8 - speed8) : speed6;
      tot_u += !obstacles[ii + (jj - 1)*params.nx] ? sqrtf(u_sq) : 0.00f;
    }
  }
  return tot_u / (float)(params.global_cells);
}

float av_velocity(const t_param params, float* restrict cells_s0, float* restrict cells_s1, float* restrict cells_s2, 
                                   float* restrict cells_s3, float* restrict cells_s4, float* restrict cells_s5, 
                                   float* restrict cells_s6, float* restrict cells_s7, float* restrict cells_s8, int* restrict obstacles)
{
  int    tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;
  __assume_aligned(obstacles, 64);
  __assume_aligned(cells_s0, 64);
  __assume_aligned(cells_s1, 64);
  __assume_aligned(cells_s2, 64);
  __assume_aligned(cells_s3, 64);
  __assume_aligned(cells_s4, 64);
  __assume_aligned(cells_s5, 64);
  __assume_aligned(cells_s6, 64);
  __assume_aligned(cells_s7, 64);
  __assume_aligned(cells_s8, 64);
  __assume((params.nx) % 2 == 0);
  __assume((params.nx) % 4 == 0);
  __assume((params.nx) % 8 == 0);
  __assume((params.nx) % 16 == 0);
  /* loop over all non-blocked cells */
  for (int jj = 1; jj < params.ny + 1; jj++)
  { 
    #pragma omp simd
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* ignore occupied cells */
      if (!obstacles[ii + (jj - 1)*params.nx])
      {
        /* local density total */
        float local_density = 0.f;

        local_density = cells_s0[ii + jj*params.nx] + cells_s1[ii + jj*params.nx]
                      + cells_s2[ii + jj*params.nx] + cells_s3[ii + jj*params.nx]
                      + cells_s4[ii + jj*params.nx] + cells_s5[ii + jj*params.nx]
                      + cells_s6[ii + jj*params.nx] + cells_s7[ii + jj*params.nx]
                      + cells_s8[ii + jj*params.nx];
        local_density = 1 / local_density;
        /* x-component of velocity */
        float u_x = (cells_s1[ii + jj*params.nx]
                      + cells_s5[ii + jj*params.nx]
                      + cells_s8[ii + jj*params.nx]
                      - cells_s3[ii + jj*params.nx]
                         - cells_s6[ii + jj*params.nx]
                         - cells_s7[ii + jj*params.nx])
                     * local_density;
        /* compute y velocity component */
        float u_y = (cells_s2[ii + jj*params.nx]
                      + cells_s5[ii + jj*params.nx]
                      + cells_s6[ii + jj*params.nx]
                      - cells_s4[ii + jj*params.nx]
                         - cells_s7[ii + jj*params.nx]
                         - cells_s8[ii + jj*params.nx])
                     * local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
      }
    }
  }

  return tot_u / (float)(params.global_cells);
}

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed* cells_ptr, t_speed* tmp_cells_ptr,
               t_speed* send_buffer, t_speed* receive_buffer,
               int** obstacles_ptr, float** av_vels_ptr,
               float** av_vels_buffer_ptr, int rank, int size)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->global_nx));

  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->global_ny));

  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));

  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));

  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  int rows = params->global_ny/size;
  int rows_rem = params->global_ny%size;
  params->nx = params->global_nx;
  int start_row = rank * rows;
  rows += (rank < rows_rem) ? 1 : 0;
  params->ny = rows;
  int cumul_rem = (rank < rows_rem) ? rank : rows_rem;
  start_row += (rank == 0) ? 0 : cumul_rem;
  int end_row = start_row + rows - 1;
  params->start_row = start_row;
  params->end_row = end_row;

  /* main grid */
  (*cells_ptr).s0 = (float*)_mm_malloc(sizeof(float) * ((params->ny + 2) * params->nx), 64);
  (*cells_ptr).s1 = (float*)_mm_malloc(sizeof(float) * ((params->ny + 2) * params->nx), 64);
  (*cells_ptr).s2 = (float*)_mm_malloc(sizeof(float) * ((params->ny + 2) * params->nx), 64);
  (*cells_ptr).s3 = (float*)_mm_malloc(sizeof(float) * ((params->ny + 2) * params->nx), 64);
  (*cells_ptr).s4 = (float*)_mm_malloc(sizeof(float) * ((params->ny + 2) * params->nx), 64);
  (*cells_ptr).s5 = (float*)_mm_malloc(sizeof(float) * ((params->ny + 2) * params->nx), 64);
  (*cells_ptr).s6 = (float*)_mm_malloc(sizeof(float) * ((params->ny + 2) * params->nx), 64);
  (*cells_ptr).s7 = (float*)_mm_malloc(sizeof(float) * ((params->ny + 2) * params->nx), 64);
  (*cells_ptr).s8 = (float*)_mm_malloc(sizeof(float) * ((params->ny + 2) * params->nx), 64);

  if ((*cells_ptr).s0 == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);
  if ((*cells_ptr).s1 == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);
  if ((*cells_ptr).s2 == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);
  if ((*cells_ptr).s3 == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);
  if ((*cells_ptr).s4 == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);
  if ((*cells_ptr).s5 == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);
  if ((*cells_ptr).s6 == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);
  if ((*cells_ptr).s7 == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);
  if ((*cells_ptr).s8 == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  (*tmp_cells_ptr).s0 = (float*)_mm_malloc(sizeof(float) * ((params->ny + 2) * params->nx), 64);
  (*tmp_cells_ptr).s1 = (float*)_mm_malloc(sizeof(float) * ((params->ny + 2) * params->nx), 64);
  (*tmp_cells_ptr).s2 = (float*)_mm_malloc(sizeof(float) * ((params->ny + 2) * params->nx), 64);
  (*tmp_cells_ptr).s3 = (float*)_mm_malloc(sizeof(float) * ((params->ny + 2) * params->nx), 64);
  (*tmp_cells_ptr).s4 = (float*)_mm_malloc(sizeof(float) * ((params->ny + 2) * params->nx), 64);
  (*tmp_cells_ptr).s5 = (float*)_mm_malloc(sizeof(float) * ((params->ny + 2) * params->nx), 64);
  (*tmp_cells_ptr).s6 = (float*)_mm_malloc(sizeof(float) * ((params->ny + 2) * params->nx), 64);
  (*tmp_cells_ptr).s7 = (float*)_mm_malloc(sizeof(float) * ((params->ny + 2) * params->nx), 64);
  (*tmp_cells_ptr).s8 = (float*)_mm_malloc(sizeof(float) * ((params->ny + 2) * params->nx), 64);

  if ((*tmp_cells_ptr).s0 == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);
  if ((*tmp_cells_ptr).s1 == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);
  if ((*tmp_cells_ptr).s2 == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);
  if ((*tmp_cells_ptr).s3 == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);
  if ((*tmp_cells_ptr).s4 == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);
  if ((*tmp_cells_ptr).s5 == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);
  if ((*tmp_cells_ptr).s6 == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);
  if ((*tmp_cells_ptr).s7 == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);
  if ((*tmp_cells_ptr).s8 == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  (*send_buffer).s0 = (float*)_mm_malloc(sizeof(float) * (params->nx), 64);
  (*send_buffer).s1 = (float*)_mm_malloc(sizeof(float) * (params->nx), 64);
  (*send_buffer).s2 = (float*)_mm_malloc(sizeof(float) * (params->nx), 64);
  (*send_buffer).s3 = (float*)_mm_malloc(sizeof(float) * (params->nx), 64);
  (*send_buffer).s4 = (float*)_mm_malloc(sizeof(float) * (params->nx), 64);
  (*send_buffer).s5 = (float*)_mm_malloc(sizeof(float) * (params->nx), 64);
  (*send_buffer).s6 = (float*)_mm_malloc(sizeof(float) * (params->nx), 64);
  (*send_buffer).s7 = (float*)_mm_malloc(sizeof(float) * (params->nx), 64);
  (*send_buffer).s8 = (float*)_mm_malloc(sizeof(float) * (params->nx), 64);

  if ((*send_buffer).s0 == NULL) die("cannot allocate memory for send_buffer", __LINE__, __FILE__);
  if ((*send_buffer).s1 == NULL) die("cannot allocate memory for send_buffer", __LINE__, __FILE__);
  if ((*send_buffer).s2 == NULL) die("cannot allocate memory for send_buffer", __LINE__, __FILE__);
  if ((*send_buffer).s3 == NULL) die("cannot allocate memory for send_buffer", __LINE__, __FILE__);
  if ((*send_buffer).s4 == NULL) die("cannot allocate memory for send_buffer", __LINE__, __FILE__);
  if ((*send_buffer).s5 == NULL) die("cannot allocate memory for send_buffer", __LINE__, __FILE__);
  if ((*send_buffer).s6 == NULL) die("cannot allocate memory for send_buffer", __LINE__, __FILE__);
  if ((*send_buffer).s7 == NULL) die("cannot allocate memory for send_buffer", __LINE__, __FILE__);
  if ((*send_buffer).s8 == NULL) die("cannot allocate memory for send_buffer", __LINE__, __FILE__);

  (*receive_buffer).s0 = (float*)_mm_malloc(sizeof(float) * (params->nx), 64);
  (*receive_buffer).s1 = (float*)_mm_malloc(sizeof(float) * (params->nx), 64);
  (*receive_buffer).s2 = (float*)_mm_malloc(sizeof(float) * (params->nx), 64);
  (*receive_buffer).s3 = (float*)_mm_malloc(sizeof(float) * (params->nx), 64);
  (*receive_buffer).s4 = (float*)_mm_malloc(sizeof(float) * (params->nx), 64);
  (*receive_buffer).s5 = (float*)_mm_malloc(sizeof(float) * (params->nx), 64);
  (*receive_buffer).s6 = (float*)_mm_malloc(sizeof(float) * (params->nx), 64);
  (*receive_buffer).s7 = (float*)_mm_malloc(sizeof(float) * (params->nx), 64);
  (*receive_buffer).s8 = (float*)_mm_malloc(sizeof(float) * (params->nx), 64);

  if ((*receive_buffer).s0 == NULL) die("cannot allocate memory for receive_buffer", __LINE__, __FILE__);
  if ((*receive_buffer).s1 == NULL) die("cannot allocate memory for receive_buffer", __LINE__, __FILE__);
  if ((*receive_buffer).s2 == NULL) die("cannot allocate memory for receive_buffer", __LINE__, __FILE__);
  if ((*receive_buffer).s3 == NULL) die("cannot allocate memory for receive_buffer", __LINE__, __FILE__);
  if ((*receive_buffer).s4 == NULL) die("cannot allocate memory for receive_buffer", __LINE__, __FILE__);
  if ((*receive_buffer).s5 == NULL) die("cannot allocate memory for receive_buffer", __LINE__, __FILE__);
  if ((*receive_buffer).s6 == NULL) die("cannot allocate memory for receive_buffer", __LINE__, __FILE__);
  if ((*receive_buffer).s7 == NULL) die("cannot allocate memory for receive_buffer", __LINE__, __FILE__);
  if ((*receive_buffer).s8 == NULL) die("cannot allocate memory for receive_buffer", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(int) * (params->ny * params->nx));

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density      / 9.f;
  float w2 = params->density      / 36.f;

  for (int jj = 1; jj < params->ny + 1; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      /* centre */
      (*cells_ptr).s0[ii + jj*params->nx] = w0;
      /* axis directions */
      (*cells_ptr).s1[ii + jj*params->nx] = w1;
      (*cells_ptr).s2[ii + jj*params->nx] = w1;
      (*cells_ptr).s3[ii + jj*params->nx] = w1;
      (*cells_ptr).s4[ii + jj*params->nx] = w1;
      /* diagonals */
      (*cells_ptr).s5[ii + jj*params->nx] = w2;
      (*cells_ptr).s6[ii + jj*params->nx] = w2;
      (*cells_ptr).s7[ii + jj*params->nx] = w2;
      (*cells_ptr).s8[ii + jj*params->nx] = w2;
    }
  }

  for (int jj = 1; jj < params->ny + 1; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      /* centre */
      (*tmp_cells_ptr).s0[ii + jj*params->nx] = 0.00f;
      /* axis directions */
      (*tmp_cells_ptr).s1[ii + jj*params->nx] = 0.00f;
      (*tmp_cells_ptr).s2[ii + jj*params->nx] = 0.00f;
      (*tmp_cells_ptr).s3[ii + jj*params->nx] = 0.00f;
      (*tmp_cells_ptr).s4[ii + jj*params->nx] = 0.00f;
      /* diagonals */
      (*tmp_cells_ptr).s5[ii + jj*params->nx] = 0.00f;
      (*tmp_cells_ptr).s6[ii + jj*params->nx] = 0.00f;
      (*tmp_cells_ptr).s7[ii + jj*params->nx] = 0.00f;
      (*tmp_cells_ptr).s8[ii + jj*params->nx] = 0.00f;
    }
  }

  /* first set all cells in obstacle array to zero */
  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      (*obstacles_ptr)[ii + jj*params->nx] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->global_nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->global_ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);
    /* assign to array */
    if (yy >= params->start_row && yy <= params->end_row) {
      int diff = (rank == 0) ? 0 : params->start_row;
      (*obstacles_ptr)[xx + (yy - diff)*params->nx] = blocked;
    }
  }

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);
  if (rank == 0){
    *av_vels_buffer_ptr = (float*)malloc(sizeof(float) * params->maxIters);
  }

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed* cells_ptr, t_speed* tmp_cells_ptr,
             t_speed* send_buffer, t_speed* receive_buffer,
             int** obstacles_ptr, float** av_vels_ptr, float** av_vels_buffer_ptr, int rank)
{
  /*
  ** free up allocated memory
  */
  _mm_free((*cells_ptr).s0);
  (*cells_ptr).s0 = NULL;
  _mm_free((*cells_ptr).s1);
  (*cells_ptr).s1 = NULL;
  _mm_free((*cells_ptr).s2);
  (*cells_ptr).s2 = NULL;
  _mm_free((*cells_ptr).s3);
  (*cells_ptr).s3 = NULL;
  _mm_free((*cells_ptr).s4);
  (*cells_ptr).s4 = NULL;
  _mm_free((*cells_ptr).s5);
  (*cells_ptr).s5 = NULL;
  _mm_free((*cells_ptr).s6);
  (*cells_ptr).s6 = NULL;
  _mm_free((*cells_ptr).s7);
  (*cells_ptr).s7 = NULL;
  _mm_free((*cells_ptr).s8);
  (*cells_ptr).s8 = NULL;

  _mm_free((*tmp_cells_ptr).s0);
  (*tmp_cells_ptr).s0 = NULL;
  _mm_free((*tmp_cells_ptr).s1);
  (*tmp_cells_ptr).s1 = NULL;
  _mm_free((*tmp_cells_ptr).s2);
  (*tmp_cells_ptr).s2 = NULL;
  _mm_free((*tmp_cells_ptr).s3);
  (*tmp_cells_ptr).s3 = NULL;
  _mm_free((*tmp_cells_ptr).s4);
  (*tmp_cells_ptr).s4 = NULL;
  _mm_free((*tmp_cells_ptr).s5);
  (*tmp_cells_ptr).s5 = NULL;
  _mm_free((*tmp_cells_ptr).s6);
  (*tmp_cells_ptr).s6 = NULL;
  _mm_free((*tmp_cells_ptr).s7);
  (*tmp_cells_ptr).s7 = NULL;
  _mm_free((*tmp_cells_ptr).s8);
  (*tmp_cells_ptr).s8 = NULL;

  _mm_free((*send_buffer).s0);
  (*send_buffer).s0 = NULL;
  _mm_free((*send_buffer).s1);
  (*send_buffer).s1 = NULL;
  _mm_free((*send_buffer).s2);
  (*send_buffer).s2 = NULL;
  _mm_free((*send_buffer).s3);
  (*send_buffer).s3 = NULL;
  _mm_free((*send_buffer).s4);
  (*send_buffer).s4 = NULL;
  _mm_free((*send_buffer).s5);
  (*send_buffer).s5 = NULL;
  _mm_free((*send_buffer).s6);
  (*send_buffer).s6 = NULL;
  _mm_free((*send_buffer).s7);
  (*send_buffer).s7 = NULL;
  _mm_free((*send_buffer).s8);
  (*send_buffer).s8 = NULL;

  _mm_free((*receive_buffer).s0);
  (*receive_buffer).s0 = NULL;
  _mm_free((*receive_buffer).s1);
  (*receive_buffer).s1 = NULL;
  _mm_free((*receive_buffer).s2);
  (*receive_buffer).s2 = NULL;
  _mm_free((*receive_buffer).s3);
  (*receive_buffer).s3 = NULL;
  _mm_free((*receive_buffer).s4);
  (*receive_buffer).s4 = NULL;
  _mm_free((*receive_buffer).s5);
  (*receive_buffer).s5 = NULL;
  _mm_free((*receive_buffer).s6);
  (*receive_buffer).s6 = NULL;
  _mm_free((*receive_buffer).s7);
  (*receive_buffer).s7 = NULL;
  _mm_free((*receive_buffer).s8);
  (*receive_buffer).s8 = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  if (rank == 0) {
    free(*av_vels_buffer_ptr);
    *av_vels_buffer_ptr = NULL;
  }

  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, t_speed cells, int* obstacles, float av_vel)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return av_vel * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speed cells)
{
  float total = 0.f;  /* accumulator */

  for (int jj = 1; jj < params.ny + 1; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      total += cells.s0[ii + jj*params.nx] + cells.s1[ii + jj*params.nx]
                      + cells.s2[ii + jj*params.nx] + cells.s3[ii + jj*params.nx]
                      + cells.s4[ii + jj*params.nx] + cells.s5[ii + jj*params.nx]
                      + cells.s6[ii + jj*params.nx] + cells.s7[ii + jj*params.nx]
                      + cells.s8[ii + jj*params.nx];
    }
  }

  return total;
}

int write_values(const t_param params, t_speed cells, int* obstacles, float* av_vels, int rank, int size)
{
  FILE* fp;                     /* file pointer */
  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  for (int proc = 0; proc < size; proc++) {
    if (rank == proc) {
      if (rank == 0) fp = fopen(FINALSTATEFILE, "w");
      else fp = fopen(FINALSTATEFILE, "a");
      if (fp == NULL)
      {
        die("could not open file output file", __LINE__, __FILE__);
      }
      for (int jj = 1; jj < params.ny + 1; jj++)
      {
        for (int ii = 0; ii < params.nx; ii++)
        {
          /* an occupied cell */
          if (obstacles[ii + (jj - 1)*params.nx])
          {
            u_x = u_y = u = 0.f;
            pressure = params.density * c_sq;
          }
          /* no obstacle */
          else
          {
            local_density = 0.f;
            local_density = cells.s0[ii + jj*params.nx] + cells.s1[ii + jj*params.nx]
                          + cells.s2[ii + jj*params.nx] + cells.s3[ii + jj*params.nx]
                          + cells.s4[ii + jj*params.nx] + cells.s5[ii + jj*params.nx]
                          + cells.s6[ii + jj*params.nx] + cells.s7[ii + jj*params.nx]
                          + cells.s8[ii + jj*params.nx];

            /* compute x velocity component */
            u_x = (cells.s1[ii + jj*params.nx]
                  + cells.s5[ii + jj*params.nx]
                  + cells.s8[ii + jj*params.nx]
                  - (cells.s3[ii + jj*params.nx]
                      + cells.s6[ii + jj*params.nx]
                      + cells.s7[ii + jj*params.nx]))
                  / local_density;
            /* compute y velocity component */
            u_y = (cells.s2[ii + jj*params.nx]
                  + cells.s5[ii + jj*params.nx]
                  + cells.s6[ii + jj*params.nx]
                  - (cells.s4[ii + jj*params.nx]
                      + cells.s7[ii + jj*params.nx]
                      + cells.s8[ii + jj*params.nx]))
                  / local_density;
            /* compute norm of velocity */
            u = sqrtf((u_x * u_x) + (u_y * u_y));
            /* compute pressure */
            pressure = local_density * c_sq;
          }

          /* write to file */
          fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, (jj + params.start_row - 1), u_x, u_y, u, pressure, obstacles[ii + (jj - 1)*params.nx]);
        }
      }
      fclose(fp);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  if (rank == 0){
    fp = fopen(AVVELSFILE, "w");

    if (fp == NULL)
    {
      die("could not open file output file", __LINE__, __FILE__);
    }

    for (int ii = 0; ii < params.maxIters; ii++)
    {
      fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
    }

    fclose(fp);
  }

  return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}