/*
  Copyright (C) 2022  Light and Molecules Group

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

/*
  # Implementation of a parser for the ``.cis`` files from ORCA.

  ## Purpose
  
  In Newton-X we need information about the singles amplitudes computed from
  TD-DFT (or TDA) / CIS.  This information is written on disk in a file with
  format ``.cis``.  Here we define a function ``orca_parse_cis`` that take the
  filename as argument, and returns:

  - A set of files ``orca.cis_rN.dat`` containing the amplitudes as a vector, in
    order (i, a) ;
  - A file ``energies.dat`` containing the transition energies for each root.

  ## Implementation
  
  The ``cis`` file from Orca has the following structure:

  1. A general header containing 9 ``int`` corresponding, in order, to :

     - the number of vectors written ;
     - the index of the first (non-frozen) alpha occupied orbital ;
     - the index of highest occupied alpha orbital ;
     - the index of the lowest virtual alpha orbital ;
     - the index of the last (non discarded) alpha virtual orbital ;
     - the index of the first (non-frozen) beta occupied orbital ;
     - the index of highest occupied beta orbital ;
     - the index of the lowest virtual beta orbital ;
     - the index of the last (non discarded) beta virtual orbital ;

  2. The set of vectors, correponding to the excitations computed.  Each vector
     has the following structure:

     - the number of amplitudes computed in the vector (``int``) ;
     - the symmetry (``int``) ;
     - the spin-multiplicity (``int``) ;
     - a parameter ``iblock`` (``int``, unused) ;
     - the root number (``int``) ;
     - the energy of the root (``double``) ;
     - a ``bool`` indicating if the energy is a transition energy or a total energy.

  ## Notes

  - The header of the vectors are serialized and written as ``char``.

  - In TDDFT, for each excitation, we have 2 vectors printed in sequence, with the
    first corresponding to X+Y, and the second to X-Y.

  - As of version 5.0.3, the root numbering / energies is buggy.  The energies are printed
    in order, irrespective of the X+Y or X-Y character.  The root number has the same problem.
    In our implementation, we print ALL vectors, and only select the one we need in the
    NX interface (see ``mod_orca.f90``).  For the energies, we print all energies, and
    read only the first N.
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

/*
File descriptor for an ORCA cis file.
*/
typedef struct orca_cis_file_t {
  char name[FILENAME_MAX + 1];
  // Filename.
  FILE *fh;
  // File handler.
  int opened;
  // Indicate if the file is opened or not.
  int nvec;
  // Number of vectors printed in the file.
  int header[8];
  // Orbital window.
  //
  // - `header[0]`: Lowest non-frozen (alpha) occupied ;
  // - `header[1]`: Highest (alpha) occupied;
  // - `header[2]`: Lowest (alpha) virtual;
  // - `header[3]`: Highest (alpha) virtual;
  //
  // `header[4]` to `header[7]` are similar, but for beta orbitals.
  int status;
  // Error status;
  int nocc[2];
  // Number of occupied orbitals.
  int nvirt[2];
  // Number of virtual orbitals.
  int nele[2];
  // Number of singles amplitudes that should be printed.
  
} Orca_Cis_File;

/*
  General information about a CIS vector.
*/
typedef struct orca_cis_vec_header_t {
  int n;
  // Number of elements (amplitudes) in the vector.
  int sym;
  // Symmetry (not used).
  int mult;
  // Spin multiplicity.
  int iblock;
  // ?
  int iroot;
  // Root number (not reliable in Orca 5.0.3).
  double en;
  // Energy of the transtion.
  bool is_transition;
  // Is the energy a transition energy (1), or the total energy (0)?

} Orca_Cis_Vec_Header;

typedef struct orca_cis_vec_t {
  Orca_Cis_Vec_Header header;
  double *sampl;
  // Array containing the single amplitudes.
} Orca_Cis_Vec;


/*
  Open an ORCA cis file.

  The function also reads the header of the file, and populates a struct
  ``Orca_Cis_File``.
 */
int orca_open_file(char *filename, Orca_Cis_File *cis_file) {

  cis_file->name[0] = '\0';
  strncat(cis_file->name, filename, sizeof(cis_file->name) - 1);
  cis_file->fh = fopen(filename, "rb");

  if (cis_file->fh == NULL) {
    cis_file->status = -1;
    cis_file->opened = 0;
    return -1;
  }

  int n = 0;
  cis_file->opened = 1; 
  n = fread(&cis_file->nvec, sizeof(int), 1, cis_file->fh);
  n = fread(&cis_file->header, sizeof(int), 8, cis_file->fh);
  cis_file->status = 0;

  cis_file->nocc[0] = cis_file->header[1] - cis_file->header[0] + 1;
  cis_file->nvirt[0] = cis_file->header[3] - cis_file->header[2] + 1;

  cis_file->nocc[1] = -1;
  cis_file->nvirt[1] = -1;
  cis_file->nele[1] = -1;
  if (cis_file->header[4] > 0) {
    cis_file->nocc[1] = cis_file->header[5] - cis_file->header[4] + 1;
    cis_file->nvirt[1] = cis_file->header[7] - cis_file->header[6] + 1;
    cis_file->nele[1] = cis_file->nocc[1] * cis_file->nvirt[1];
  }

  cis_file->nele[0] = cis_file->nocc[0] * cis_file->nvirt[0];

  return 0;
}

/*
  Print information about an ORCA cis file.
*/
void orca_print_file(Orca_Cis_File *cis_file) {
  printf("ORCA CIS FILE: %s\n", cis_file->name);

  printf("  NVEC           = %d\n", cis_file->nvec);
  printf("  ORBITAL WINDOW =");
  for (int i = 0; i < 8; i++) {
    printf(" %d", cis_file->header[i]);
  }
  printf("\n");
  printf("  NOCC  (A)      = %d\n", cis_file->nocc[0]);
  printf("  NVIRT (A)      = %d\n", cis_file->nvirt[0]);
  printf("  NELE (A)       = %d\n", cis_file->nele[0]);
  if (cis_file->nocc[1] > 0) {
    printf("\n");
    printf("  NOCC  (B)      = %d\n", cis_file->nocc[1]);
    printf("  NVIRT (B)      = %d\n", cis_file->nvirt[1]);
    printf("  NELE (B)       = %d\n", cis_file->nele[1]);
  }
  printf("\n");
}

void orca_read_vector_header(FILE * f, Orca_Cis_Vec * vec) {
  char bytes[sizeof(Orca_Cis_Vec_Header)];
  int n = 0;
  n = fread(&bytes, sizeof(Orca_Cis_Vec_Header), 1, f);
  vec->header = *((Orca_Cis_Vec_Header*) bytes);
}

void orca_close_vector(Orca_Cis_Vec *vec) {
  free(vec->sampl);
}

void orca_print_vector_header(Orca_Cis_Vec_Header * vec) {
  printf("CIS VECTOR for root %d\n", vec->iroot);
  printf("  LENGTH     = %d\n", vec->n);
  printf("  SYMMETRY   = %d\n", vec->sym);
  printf("  MULT       = %d\n", vec->mult);
  printf("  IBLOCK     = %d\n", vec->iblock);
  printf("  ENERGY     = %20.12f\n", vec->en);
  printf("  TRANSITION ? %d\n", vec->is_transition);
}

int orca_read_vector_amplitudes(FILE *f, Orca_Cis_Vec *vec) {
  vec->sampl = (double *) malloc(vec->header.n * sizeof(double));

  int res;
  res = fread(vec->sampl, sizeof(double), vec->header.n, f);
  
  return res;
}

void orca_write_vector_amplitudes(Orca_Cis_Vec *vec, char *filename) {

  FILE *f;
  f = fopen(filename, "w");

  for (int i = 0; i < vec->header.n; i++) {
    fprintf(f, "  %20.12f\n", vec->sampl[i]);
  }
  fclose(f);
}

int orca_parse_cis(char *filename, int debug) {


  Orca_Cis_File *cis_file = malloc(sizeof(*cis_file));
  if (!cis_file) {
    printf("Memory allocation failed\n");
    return -1;
  }

  int stat = orca_open_file(filename, cis_file);
  if (stat != 0) {
    printf("Error in reading %s\n", filename);
    return -1;
  }
  if (debug > 0) {
    orca_print_file(cis_file);
  }

  double energies[cis_file->nvec];
  for (int i = 0; i < cis_file->nvec; i++) {
    Orca_Cis_Vec vec;
    orca_read_vector_header(cis_file->fh, &vec);
    energies[i] = vec.header.en;
    if (debug > 0) {
      orca_print_vector_header(&vec.header);
    }
    int res = orca_read_vector_amplitudes(cis_file->fh, &vec);

    char output[256];
    snprintf(output, 256, "%s_r%d.dat", filename, i + 1);
    orca_write_vector_amplitudes(&vec, output);
    orca_close_vector(&vec);
  }

  FILE *f = fopen("energies.dat", "w");
  for (int i = 0; i < cis_file->nvec; i++) {
    fprintf(f, "Vector %i: %20.12f\n", i, energies[i]);
  }
  fclose(f);
  fclose(cis_file->fh);
  free(cis_file);

  return 0;
}


// int main(int argc, char *argv[]) {
// 
//   char *filename;
//   filename = argv[1];
// 
//   int res = orca_parse_cis(filename, 0, 1);
// 
//   return 0;
// }
