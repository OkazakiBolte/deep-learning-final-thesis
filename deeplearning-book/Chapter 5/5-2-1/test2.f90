program test2
 implicit none

  integer :: m, j, k, i, h
!  real(8), allocatable :: w(:)
  real(8) :: w(5) = (/ 2.0_8, 0.0_8, -3.0_8, -0.5_8, 1.0_8 /)
  real(8) :: x, y, xmin, xmax, width, delta

! 何次式？
  m = 4
!  allocate(w(m + 1))
!  read(*,*) w
!  w(m + 1) = (/ 2.0_8, 0.0_8, -3.0_8, -0.5_8, 1.0_8 /)

! 分割数
  k = 100

  xmin = -2.0_8
  xmax =  2.0_8
  width = xmax - xmin
  delta = width / real(k)

!  do i = 1, m + 1
!    write(*,*) i, w(i)
!  end do

!  write(*,*) xmin, xmax, width, delta

  do j = 0, k
    x = xmin + real(j) * delta
    y = 0.0_8
    do i = 1, m + 1
      y = y + w(i) * x ** (real(i - 1))
!      write(*,*) i, x, y
    end do
    write(*,*) j, x, y
  end do

!  deallocate(w)

 stop
end program
