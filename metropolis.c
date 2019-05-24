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
#define L 128
#define N (L*L)
#define XNN 1
#define YNN L
#define nTemps 25
#define dT 0.05
#define Tmin 1.625
#define nSamples 10
#define eqSteps 2000*N
#define nReset 1
#define bits 8

/* Global variables are initialized*/
int s[N];
unsigned char s_byte[(int)N/bits];
int powers[bits];
double acceptances[5];
double betas[nTemps];
char file_name[100] = "/home/dg/Documents/Monograph/Saved/size128/nConfigs2500/spins.bin"; // "spins.bin";
char file_name_csv[60] = "spins.csv";

/* Functions' headers*/
double randu();
void initialize_powers();
void initialize_betas();
void initialize_lattice();
void set_acceptances(double beta);
void sweep_lattice(int steps);
void spins2byte(int k);
void export_lattice();
void export_lattice_as_csv();

/* Functions' bodies*/
void main() {
  /* Initialize array of betas */
  initialize_powers();
  initialize_betas(); 

  // for (int m = 0; m<bits; m++) printf("Power %i: %u...\n", m, powers[m]);
  // for (int m = 0; m<nTemps; m++) printf("Beta %i: %.4f...\n", m, betas[m]);

  int t, k;
  for (t=0; t<nTemps; t++) {
    printf("\rSampling lattices for temperature %.3f...", 1/betas[t]);
    fflush(stdout);

    /* Initialize lattice array */
    initialize_lattice();

    /* Calculate acceptance acceptancesabilities for positive costs */
    set_acceptances(betas[t]);
    // for (int m = 0; m<5; m++) printf("A%i: %.3f...\n", m, acceptances[m]);

    /* Carry out equilibration */
    sweep_lattice(eqSteps);

    /* Sample lattices from Boltzmann distribution and store on file*/
    for (k=0; k<nSamples; k++) {
      sweep_lattice(N);
      spins2byte(k);
      // export_lattice();
      // export_lattice_as_csv();

      if (((k+1) % nReset) == 0) {
        initialize_lattice();
        sweep_lattice(eqSteps);
      }
    }
  }
  printf("Finished Metropolis sampling...\n");  
}

double randu() {
  return ((double) random() / RAND_MAX);
}

void initialize_powers() {
  int b;
  powers[0] = 1;
  for (b=1; b<bits; b++) powers[b] = 2*powers[b-1];
}

void initialize_betas() {
  int i;
  for (i=0; i<nTemps; i++) betas[i] = (double) 1/(Tmin+(i*dT));
}

void initialize_lattice() {
  int n;
  for (n=0; n<N; n++) {    
    if (randu() > 0.5) s[n] = 1;
    else s[n] = -1;    
  } 
}

void set_acceptances(double beta)
{
  int i;
  for (i=2; i<5; i+=2) acceptances[i] = (double) exp(-2*beta*i);
}

void sweep_lattice(int steps) {
  int i, k;
  int nn, sum, delta;

  for (k=0; k<steps; k++) {

    /* Choose a site */
    i = (int) floor(N*randu());
    // printf("\r Chosen spin %i in step %i...", i, k);

    /* Calculate the sum of the neighbouring spins*/
    if ((nn=i+XNN)>=N) nn -= N; 
    sum = s[nn];
    if ((nn=i-XNN)<0) nn += N; 
    sum += s[nn];
    if ((nn=i+YNN)>=N) nn -= N;  
    sum += s[nn];
    if ((nn=i-YNN)<0) nn += N; 
    sum += s[nn];
    // printf("\r Neighbour sum: %i...", sum);

    /* Calculate the change in energy */
    delta = sum*s[i];
    // printf("Energy variation: %i...\n", 2*delta);

    /* Decide whether to flip spin */
    if (delta<=0) {
      s[i] = -s[i];      
    }
    else if (randu()<acceptances[delta]) {
      s[i] = -s[i];
    }  
  }
}

void spins2byte(int n) {
  int sp = 0;
  int p = bits-1;
	for(int k = 0; k < N; k++){
    if (s[k]>0) sp += powers[p];
    p -= 1;
    if (((k+1) % bits) == 0){
      // if ((k < bits) && n == 0) printf("%u\n", sp);
      s_byte[(int)(k/bits)] = sp;
      sp = 0;
      p = bits-1;
    } 
	}
}

void export_lattice() {	
	FILE *file = fopen(file_name, "ab+");

  fwrite(s_byte, sizeof(s_byte), 1, file);

  // unsigned int sp = 0;
	// for(int k = 0; k < N; k++){
  //   if (s[k]>0) sp = (2*sp)+1;
  //   else sp = 2*sp;
  //   if (((k+1) % 32) == 0){
  //     if (k < N-1){
  //       fprintf(file, "%i,", sp);
  //       // printf("%u, %i\n", sp, sp);
  //       // for (int j = 0; j < 1000000; j++);
  //     }
  //     else fprintf(file, "%i\n", sp);
  //     sp = 0;
  //   } 
	// }

  // for(int k = 0; k < N-1; k++){
  //   fprintf(file, "%i,", s[k]);
  // }
  // fprintf(file, "%i\n", s[N-1]);

	fclose(file);
}

void export_lattice_as_csv() {	
	FILE *file = fopen(file_name_csv, "ab+");
  for(int k = 0; k < N-1; k++){
    fprintf(file, "%i,", s[k]);
  }
  fprintf(file, "%i\n", s[N-1]);

	fclose(file);
}
