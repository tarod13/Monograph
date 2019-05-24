/* betas[] = array of inverse temperatures
 * acceptances[] = array of acceptance acceptancesabilities
 * s[] = lattice of spins with helical boundary conditions
 * L = constant edge lenght of lattice
 * N = number of spins
 */

/* Needed libraries are included */
#include <stdio.h>
#include <stdlib.h> //required for random()
#include <string.h>
#include <math.h>

/* Global required constants are defined */
#define L 64
#define N (L*L)
#define XNN 1
#define YNN L
#define nTemps 28
#define dT 0.05
#define Tmin 0.225 //0.225
#define Tmax 1.625
#define nSamples 100
#define eqSteps 40000*N
#define nReset 25
#define TPI 6.28318530718
#define cPI 2.19911485751
#define Boundary 1 /*0 if periodic, 1 if not*/

/* Global variables are initialized*/
float s[N];
float betas[nTemps];
char file_name[100] = "/home/dg/Documents/Monograph/Saved/XY/size64/nConfigs14000/spins_not_periodic_2.bin"; // "spins.bin";
// char file_name_csv[60] = "spins.csv";

/* Functions' headers*/
float randu();
void initialize_betas();
void initialize_lattice();
void sweep_lattice(int steps, int t);
void export_lattice();
// void export_lattice_as_csv();

/* Functions' bodies*/
void main() {
  /* Initialize array of betas */
  initialize_betas(); 

  // for (int m = 0; m<bits; m++) printf("Power %i: %u...\n", m, powers[m]);
  // for (int m = 0; m<nTemps; m++) printf("Beta %i: %.4f...\n", m, betas[m]);

  int t, k;
  for (t=0; t<nTemps; t++) {
    printf("\rSampling lattices for temperature %.3f...", 1/betas[t]);
    fflush(stdout);
    // printf("Sampling lattices for temperature %.3f...\n", 1/betas[t]);

    /* Initialize lattice array */
    initialize_lattice();

    /* Carry out equilibration */
    sweep_lattice(eqSteps, t);

    /* Sample lattices from Boltzmann distribution and store on file*/
    for (k=0; k<nSamples; k++) {
      sweep_lattice(800*N, t);
      export_lattice();      

      if (((k+1) % nReset) == 0) {
        initialize_lattice();
        sweep_lattice(eqSteps, t);
      }
    }
  }
  printf("Finished Metropolis sampling...\n");  
}

float randu() {
  return ((float) random() / RAND_MAX);
}

void initialize_betas() {
  int i;
  for (i=0; i<nTemps; i++) betas[i] = (float) 1/(Tmin+(i*dT));
}

void initialize_lattice() {
  int n;
  for (n=0; n<N; n++) {    
    s[n] = randu()*TPI;
  } 
}

void sweep_lattice(int steps, int t) {
  int i, k;
  int nn;
  float new_theta, delta;

//   int accepted = 0;

  float factor = t*dT + 0.03;
  factor /= Tmax - Tmin + 0.03;
  factor = pow(factor, 0.5);

  for (k=0; k<steps; k++) {
    delta = 0.0;

    /* Choose a site */
    i = (int) floor(N*randu());
    // printf("\r Chosen spin %i in step %i...", i, k);

    /* Choose a new direction*/
    new_theta = s[i] + (randu()-0.5)*cPI*factor;
    if (new_theta >= TPI) new_theta -= TPI;
    else if (new_theta < 0) new_theta += TPI;

    /* Calculate the energy of the current bonds*/
    if (Boundary == 0){
      if ((nn=i+XNN)>=N) nn -= N; 
      delta += cos(s[i]-s[nn]);
      if ((nn=i-XNN)<0) nn += N; 
      delta += cos(s[i]-s[nn]);
      if ((nn=i+YNN)>=N) nn -= N;  
      delta += cos(s[i]-s[nn]);
      if ((nn=i-YNN)<0) nn += N; 
      delta += cos(s[i]-s[nn]);
    }
    else{
      if ((i+1) % L != 0){
        if ((nn=i+XNN)>=N) nn -= N; 
        delta += cos(s[i]-s[nn]);
      }
      if (i % L != 0){
        if ((nn=i-XNN)<0) nn += N; 
        delta += cos(s[i]-s[nn]);
      }
      if (i < N-L){
        if ((nn=i+YNN)>=N) nn -= N;  
        delta += cos(s[i]-s[nn]);
      }
      if (i > L){
        if ((nn=i-YNN)<0) nn += N; 
        delta += cos(s[i]-s[nn]);
      }
    }

    /* Calculate the energy of the bonds after the transition*/
    if (Boundary == 0){
      if ((nn=i+XNN)>=N) nn -= N; 
      delta -= cos(new_theta-s[nn]);
      if ((nn=i-XNN)<0) nn += N; 
      delta -= cos(new_theta-s[nn]);
      if ((nn=i+YNN)>=N) nn -= N;  
      delta -= cos(new_theta-s[nn]);
      if ((nn=i-YNN)<0) nn += N; 
      delta -= cos(new_theta-s[nn]);
    }
    else{
      if ((i+1) % L != 0){
        if ((nn=i+XNN)>=N) nn -= N; 
        delta -= cos(new_theta-s[nn]);
      }
      if (i % L != 0){
        if ((nn=i-XNN)<0) nn += N; 
        delta -= cos(new_theta-s[nn]);
      }
      if (i < N-L){
        if ((nn=i+YNN)>=N) nn -= N;  
        delta -= cos(new_theta-s[nn]);
      }
      if (i > L){
        if ((nn=i-YNN)<0) nn += N; 
        delta -= cos(new_theta-s[nn]);
      }
    }
    // printf("\r Neighbour sum: %i...", sum);

    /* Calculate the change in energy */
    delta = betas[t]*delta;
    // printf("Energy variation: %i...\n", 2*delta);

    /* Decide whether to flip spin */
    if (delta<=0) {
      s[i] = new_theta;   
    //   accepted ++;
    }
    else if (randu()<exp(-delta)) {
      s[i] = new_theta;
    //   accepted ++;
    }  
  }

//   printf("Accepted %i transitions...\n", accepted);
}

void export_lattice() {	
	FILE *file = fopen(file_name, "ab+");
    fwrite(s, sizeof(s), 1, file);
	fclose(file);
}

// void export_lattice_as_csv() {	
// 	FILE *file = fopen(file_name_csv, "ab+");
//   for(int k = 0; k < N-1; k++){
//     fprintf(file, "%i,", s[k]);
//   }
//   fprintf(file, "%i\n", s[N-1]);

// 	fclose(file);
// }
