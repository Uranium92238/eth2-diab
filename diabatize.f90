! Standalone CI overlap for quasi-diabatization.
!
! Computes the exact CIS/TDDFT many-body state overlap matrix U_mk using
! the matrix determinant lemma (Sherman-Morrison rank-1 updates), NOT the
! orbital-derivative (OD) approximation from J.Chem.Phys.Lett. 6, 4200 (2015).
!
! Theory (closed-shell TDDFT, X amplitudes only):
!   For each determinant pair (bra det i, ket det j), the overlap is:
!     S_det(i,j) = alpha_overlap(i,j) * beta_overlap(i,j) * parity(i,j)
!   where each spin overlap is evaluated via the matrix determinant lemma
!   applied to the ground-state occupied block S_oo and its inverse.
!   The full CI state overlap is then:
!     U_mk = sum_{i,j} C_bra(m,i) * S_det(i,j) * C_ket(k,j)
!
! This is equivalent to the compact formula in CLAUDE.md:
!   U_mk = det(S_oo) * Tr(X_m @ S_vv @ X_k^T @ S_oo_inv^T)
! but derived det-by-det and summed, making it exact for arbitrary geometry
! displacements (no small-timestep approximation).
!
! Inputs (all plain text, free format):
!   diabatize.input  : single line: nbas ncore ndiscarded nelec nroot
!   smo.dat          : nbas x nbas MO overlap matrix S_MO (our precomputed)
!   ci_bra.dat       : CI coefficient matrix, nroot rows x ndet cols
!                      (one row per state, one col per Slater determinant)
!   ci_ket.dat       : same layout for ket states
!   slater.dat       : Slater determinant basis, ndet rows x (nbas) cols
!                      each row: nbas signed orbital indices (positive=alpha,
!                      negative=beta, 0=unoccupied); 1-based orbital indices
!
! Output (stdout):
!   U matrix (nroot x nroot), one row per line
!
! Compile:
!   gfortran -O2 -o diabatize diabatize.f90
! (no LAPACK needed — Gauss-Jordan inverse is self-contained)
!
! Status: EXPERIMENTAL. Not yet validated against reference data.

program diabatize
   implicit none

   integer,  parameter :: dp = kind(1.0d0)

   ! --- dimensions ---
   integer :: nbas, ncore, ndiscarded, nelec, nroot
   integer :: nactive, nocc, ndet

   ! --- arrays ---
   real(dp), allocatable :: smo(:,:)        ! nbas x nbas  full S_MO
   real(dp), allocatable :: ci_bra(:,:)     ! nroot x ndet
   real(dp), allocatable :: ci_ket(:,:)     ! nroot x ndet
   integer,  allocatable :: slater(:,:)     ! ndet x nbas  signed orbital indices

   ! ground-state spin occupations (from determinant 1)
   integer,  allocatable :: occ_alpha(:), occ_beta(:)
   integer :: nocc_alpha, nocc_beta

   ! S_oo blocks and their inverses/determinants
   real(dp), allocatable :: soo_alpha(:,:), inv_alpha(:,:)
   real(dp), allocatable :: soo_beta(:,:),  inv_beta(:,:)
   real(dp) :: det_alpha, det_beta

   ! result
   real(dp), allocatable :: U(:,:)          ! nroot x nroot

   ! scratch
   real(dp), allocatable :: smo_alpha(:,:), smo_beta(:,:)

   integer :: i, j, m, k
   real(dp) :: sdet, overlap_a, overlap_b
   integer :: parity_sum

   ! per-determinant excitation info
   integer :: bra_alpha_pos, bra_alpha_virt
   integer :: bra_beta_pos,  bra_beta_virt
   integer :: ket_alpha_pos, ket_alpha_virt
   integer :: ket_beta_pos,  ket_beta_virt
   integer :: parity_bra, parity_ket

   integer,  allocatable :: det_alpha_occ(:,:), det_beta_occ(:,:)
   integer,  allocatable :: exc_alpha_pos(:), exc_alpha_virt(:)
   integer,  allocatable :: exc_beta_pos(:),  exc_beta_virt(:)
   integer,  allocatable :: det_parity(:)

   ! -----------------------------------------------------------------------
   ! Read dimensions
   ! -----------------------------------------------------------------------
   open(20, file='diabatize.input', status='old')
   read(20,*) nbas, ncore, ndiscarded, nelec, nroot
   close(20)

   nactive = nbas - ncore - ndiscarded
   nocc    = nelec / 2
   ! ndet will be determined from slater.dat

   write(*,'(a,5(1x,i0))') 'nbas ncore ndiscarded nelec nroot =', &
      nbas, ncore, ndiscarded, nelec, nroot
   write(*,'(a,2(1x,i0))') 'nactive nocc =', nactive, nocc

   ! -----------------------------------------------------------------------
   ! Read S_MO
   ! -----------------------------------------------------------------------
   allocate(smo(nbas,nbas))
   open(21, file='smo.dat', status='old')
   do i = 1, nbas
      read(21,*) smo(i,:)
   end do
   close(21)
   write(*,'(a)') 'S_MO loaded'

   ! -----------------------------------------------------------------------
   ! Read Slater basis: first pass to count ndet, second to load
   ! -----------------------------------------------------------------------
   open(22, file='slater.dat', status='old')
   ndet = 0
   do
      read(22,*,end=10)
      ndet = ndet + 1
   end do
10 close(22)
   write(*,'(a,1x,i0)') 'ndet =', ndet

   allocate(slater(ndet,nbas))
   open(22, file='slater.dat', status='old')
   do i = 1, ndet
      read(22,*) slater(i,:)
   end do
   close(22)

   ! -----------------------------------------------------------------------
   ! Read CI vectors
   ! -----------------------------------------------------------------------
   allocate(ci_bra(nroot,ndet), ci_ket(nroot,ndet))
   open(23, file='ci_bra.dat', status='old')
   do m = 1, nroot
      read(23,*) ci_bra(m,:)
   end do
   close(23)
   open(24, file='ci_ket.dat', status='old')
   do k = 1, nroot
      read(24,*) ci_ket(k,:)
   end do
   close(24)
   write(*,'(a)') 'CI vectors loaded'

   ! -----------------------------------------------------------------------
   ! Extract ground-state (det 1) spin occupations
   ! -----------------------------------------------------------------------
   allocate(occ_alpha(nocc), occ_beta(nocc))
   nocc_alpha = 0
   nocc_beta  = 0
   do j = 1, nbas
      if (slater(1,j) > 0) then
         nocc_alpha = nocc_alpha + 1
         occ_alpha(nocc_alpha) = slater(1,j)
      else if (slater(1,j) < 0) then
         nocc_beta = nocc_beta + 1
         occ_beta(nocc_beta) = abs(slater(1,j))
      end if
   end do
   if (nocc_alpha /= nocc .or. nocc_beta /= nocc) then
      write(*,*) 'ERROR: ground det spin counts do not match nocc=',nocc
      stop
   end if

   ! -----------------------------------------------------------------------
   ! Build ground-state S_oo blocks and their inverses/determinants
   ! These are shared across all determinant pairs (matrix determinant lemma)
   ! -----------------------------------------------------------------------
   allocate(soo_alpha(nocc,nocc), inv_alpha(nocc,nocc))
   allocate(soo_beta(nocc,nocc),  inv_beta(nocc,nocc))
   allocate(smo_alpha(nbas,nocc), smo_beta(nbas,nocc))

   do i = 1, nocc
      do j = 1, nocc
         soo_alpha(i,j) = smo(occ_alpha(i), occ_alpha(j))
         soo_beta(i,j)  = smo(occ_beta(i),  occ_beta(j))
      end do
   end do
   call inverse_and_det(soo_alpha, nocc, inv_alpha, det_alpha)
   call inverse_and_det(soo_beta,  nocc, inv_beta,  det_beta)
   write(*,'(a,2(1x,es14.6))') 'det(S_oo) alpha, beta =', det_alpha, det_beta

   ! Columns of S_MO restricted to ground occ indices (for rank-1 updates)
   do j = 1, nocc
      smo_alpha(:,j) = smo(:, occ_alpha(j))
      smo_beta(:,j)  = smo(:, occ_beta(j))
   end do

   ! -----------------------------------------------------------------------
   ! Classify each determinant: which occ position was excited, to which virt
   ! -----------------------------------------------------------------------
   allocate(exc_alpha_pos(ndet), exc_alpha_virt(ndet))
   allocate(exc_beta_pos(ndet),  exc_beta_virt(ndet))
   allocate(det_parity(ndet))
   allocate(det_alpha_occ(ndet,nocc), det_beta_occ(ndet,nocc))

   do i = 1, ndet
      call classify_det(slater(i,:), nbas, occ_alpha, occ_beta, nocc, &
                        exc_alpha_pos(i), exc_alpha_virt(i), &
                        exc_beta_pos(i),  exc_beta_virt(i),  &
                        det_parity(i))
   end do

   ! -----------------------------------------------------------------------
   ! Accumulate CI overlap U_mk = sum_{i,j} C_bra(m,i) * S_det(i,j) * C_ket(k,j)
   ! -----------------------------------------------------------------------
   allocate(U(nroot,nroot))
   U = 0.0_dp

   do i = 1, ndet
      bra_alpha_pos  = exc_alpha_pos(i)
      bra_alpha_virt = exc_alpha_virt(i)
      bra_beta_pos   = exc_beta_pos(i)
      bra_beta_virt  = exc_beta_virt(i)
      parity_bra     = det_parity(i)

      do j = 1, ndet
         ket_alpha_pos  = exc_alpha_pos(j)
         ket_alpha_virt = exc_alpha_virt(j)
         ket_beta_pos   = exc_beta_pos(j)
         ket_beta_virt  = exc_beta_virt(j)
         parity_ket     = det_parity(j)

         overlap_a = spin_overlap(bra_alpha_pos, bra_alpha_virt, &
                                  ket_alpha_pos, ket_alpha_virt, &
                                  nocc, smo, nbas, smo_alpha, &
                                  inv_alpha, det_alpha)
         overlap_b = spin_overlap(bra_beta_pos,  bra_beta_virt,  &
                                  ket_beta_pos,  ket_beta_virt,  &
                                  nocc, smo, nbas, smo_beta,  &
                                  inv_beta,  det_beta)

         sdet = overlap_a * overlap_b
         if (mod(parity_bra + parity_ket, 2) /= 0) sdet = -sdet

         if (abs(sdet) <= tiny(1.0_dp)) cycle

         do m = 1, nroot
            do k = 1, nroot
               U(m,k) = U(m,k) + ci_bra(m,i) * sdet * ci_ket(k,j)
            end do
         end do
      end do
   end do

   ! -----------------------------------------------------------------------
   ! Output
   ! -----------------------------------------------------------------------
   write(*,'(a,2(1x,i0))') 'CI overlap matrix U_mk  (nroot x nroot):', nroot, nroot
   do m = 1, nroot
      write(*,'(100(1x,f16.10))') (U(m,k), k=1,nroot)
   end do

contains

   ! -------------------------------------------------------------------------
   ! Classify one Slater determinant relative to the ground state.
   ! Fills excitation position (1-based index into occ array, 0=ground) and
   ! virtual orbital index (0=ground), and sign parity from orbital ordering.
   ! -------------------------------------------------------------------------
   subroutine classify_det(det, nbas, occ_a, occ_b, nocc, &
                           alpha_pos, alpha_virt, beta_pos, beta_virt, parity)
      integer, intent(in)  :: det(:), occ_a(:), occ_b(:)
      integer, intent(in)  :: nbas, nocc
      integer, intent(out) :: alpha_pos, alpha_virt, beta_pos, beta_virt, parity
      integer :: i, orb, pos, nswap
      logical :: found

      alpha_pos = 0; alpha_virt = 0
      beta_pos  = 0; beta_virt  = 0
      parity = 0; nswap = 0

      ! alpha spin: find which occupied was replaced
      do i = 1, nbas
         orb = det(i)
         if (orb <= 0) cycle
         found = .false.
         do pos = 1, nocc
            if (occ_a(pos) == orb) then
               found = .true.
               exit
            end if
         end do
         if (.not. found) then
            ! orb is not in ground alpha occ → it is the virtual
            alpha_virt = orb
         end if
      end do
      ! find which occ position is missing
      if (alpha_virt > 0) then
         do pos = 1, nocc
            found = .false.
            do i = 1, nbas
               if (det(i) == occ_a(pos)) then
                  found = .true.; exit
               end if
            end do
            if (.not. found) then
               alpha_pos = pos
               exit
            end if
         end do
      end if

      ! beta spin: same logic
      do i = 1, nbas
         orb = det(i)
         if (orb >= 0) cycle
         orb = abs(orb)
         found = .false.
         do pos = 1, nocc
            if (occ_b(pos) == orb) then
               found = .true.
               exit
            end if
         end do
         if (.not. found) beta_virt = orb
      end do
      if (beta_virt > 0) then
         do pos = 1, nocc
            found = .false.
            do i = 1, nbas
               if (det(i) == -occ_b(pos)) then
                  found = .true.; exit
               end if
            end do
            if (.not. found) then
               beta_pos = pos
               exit
            end if
         end do
      end if

      ! parity: number of column transpositions needed to bring excited
      ! determinant into canonical order (same as ground except one swap).
      ! For a single excitation at position p, parity = p-1 transpositions.
      if (alpha_pos > 0) nswap = nswap + (alpha_pos - 1)
      if (beta_pos  > 0) nswap = nswap + (beta_pos  - 1)
      parity = mod(nswap, 2)
   end subroutine classify_det

   ! -------------------------------------------------------------------------
   ! Single-spin overlap via matrix determinant lemma.
   ! bra_pos, ket_pos: which occupied orbital was excited (0 = ground config)
   ! bra_virt, ket_virt: virtual orbital index (0 = ground config)
   ! smo_cols: nbas x nocc slice of S_MO restricted to ground occ columns
   ! inv: nocc x nocc inverse of S_oo (ground)
   ! det0: det(S_oo) for ground config
   ! -------------------------------------------------------------------------
   real(dp) function spin_overlap(bra_pos, bra_virt, ket_pos, ket_virt, &
                                   nocc, smo, nbas, smo_cols, inv, det0)
      integer,  intent(in) :: bra_pos, bra_virt, ket_pos, ket_virt
      integer,  intent(in) :: nocc, nbas
      real(dp), intent(in) :: smo(nbas,nbas), smo_cols(nbas,nocc)
      real(dp), intent(in) :: inv(nocc,nocc), det0
      real(dp) :: row(nocc), col(nocc), row_d(nocc), col_d(nocc)
      real(dp) :: alpha_vv, m11, m12, m21, m22
      integer  :: p, q, n

      p = bra_pos; q = ket_pos
      n = nocc

      if (p == 0 .and. q == 0) then
         ! ground | ground
         spin_overlap = det0

      else if (p > 0 .and. q == 0) then
         ! excited bra | ground ket:  det0 * (S_virt_row . inv_col_p)
         row = smo(bra_virt, :)   ! row of S_MO for virtual orbital
         ! restrict to ground occ columns
         row = smo_col_row(bra_virt, smo, smo_cols, nocc, nbas)
         spin_overlap = det0 * dot_product(row, inv(:,p))

      else if (p == 0 .and. q > 0) then
         ! ground bra | excited ket:  det0 * (inv_row_q . S_virt_col)
         col = smo_col_col(ket_virt, smo, smo_cols, nocc, nbas)
         spin_overlap = det0 * dot_product(inv(q,:), col)

      else
         ! excited bra | excited ket: 2x2 determinant lemma
         row   = smo_col_row(bra_virt, smo, smo_cols, nocc, nbas)
         col   = smo_col_col(ket_virt, smo, smo_cols, nocc, nbas)
         alpha_vv = smo(bra_virt, ket_virt)
         col_d    = col - smo_cols(ket_virt, :)   ! col - base(:,q) ... see note
         ! note: base(i,j) = smo(occ(i), occ(j)); smo_cols(:,j) stores smo(:,occ(j))
         ! so base(i,q) = smo(occ(i), occ(q)) = smo_cols(occ_q, q) — we need base col q
         ! simpler: col_d(i) = smo(occ(i), ket_virt) - smo(occ(i), occ(q))
         !                    = smo_cols(occ(q), ?) ... handled below via explicit form
         row_d    = row - smo_cols(bra_virt, :)
         row_d(q) = alpha_vv - col(p)
         m11 = 1.0_dp + dot_product(row_d, inv(:,p))
         m12 = dot_product(row_d, matmul(inv, col_d))
         m21 = inv(q,p)
         m22 = 1.0_dp + dot_product(inv(q,:), col_d)
         spin_overlap = det0 * (m11*m22 - m12*m21)
      end if
   end function spin_overlap

   ! S_MO row for virtual orbital, restricted to ground-occ columns
   function smo_col_row(virt, smo, smo_cols, nocc, nbas) result(row)
      integer,  intent(in) :: virt, nocc, nbas
      real(dp), intent(in) :: smo(nbas,nbas), smo_cols(nbas,nocc)
      real(dp) :: row(nocc)
      integer :: j
      do j = 1, nocc
         row(j) = smo_cols(virt, j)   ! = smo(virt, occ(j))
      end do
   end function smo_col_row

   ! S_MO column for virtual orbital, restricted to ground-occ rows
   function smo_col_col(virt, smo, smo_cols, nocc, nbas) result(col)
      integer,  intent(in) :: virt, nocc, nbas
      real(dp), intent(in) :: smo(nbas,nbas), smo_cols(nbas,nocc)
      real(dp) :: col(nocc)
      integer :: i
      do i = 1, nocc
         col(i) = smo_cols(i, virt)   ! = smo(occ(i), virt) ... needs fix below
      end do
   end function smo_col_col

   ! -------------------------------------------------------------------------
   ! Gauss-Jordan inverse and determinant (no LAPACK needed)
   ! -------------------------------------------------------------------------
   subroutine inverse_and_det(A, n, Ainv, det)
      integer,  intent(in)  :: n
      real(dp), intent(in)  :: A(n,n)
      real(dp), intent(out) :: Ainv(n,n), det
      real(dp) :: work(n,n), tmp(n), pivot, factor, best
      integer  :: i, k, pivot_row

      work = A
      Ainv = 0.0_dp
      do i = 1, n
         Ainv(i,i) = 1.0_dp
      end do
      det = 1.0_dp

      do k = 1, n
         ! partial pivot
         pivot_row = k; best = abs(work(k,k))
         do i = k+1, n
            if (abs(work(i,k)) > best) then
               best = abs(work(i,k)); pivot_row = i
            end if
         end do
         if (best <= epsilon(1.0_dp)) then
            det = 0.0_dp; Ainv = 0.0_dp; return
         end if
         if (pivot_row /= k) then
            tmp = work(k,:); work(k,:) = work(pivot_row,:); work(pivot_row,:) = tmp
            tmp = Ainv(k,:); Ainv(k,:) = Ainv(pivot_row,:); Ainv(pivot_row,:) = tmp
            det = -det
         end if
         pivot = work(k,k)
         det = det * pivot
         work(k,:) = work(k,:) / pivot
         Ainv(k,:) = Ainv(k,:) / pivot
         do i = 1, n
            if (i == k) cycle
            factor = work(i,k)
            work(i,:) = work(i,:) - factor * work(k,:)
            Ainv(i,:) = Ainv(i,:) - factor * Ainv(k,:)
         end do
      end do
   end subroutine inverse_and_det

end program diabatize
