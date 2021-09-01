program writeSimpleCSV
  implicit none
  integer, parameter :: n = 5
  integer i
  real :: x(n) = (/ 1.2, 0.02, 0.0003, 4.2, 5.99 /)
  real :: y(n) = (/ 9.0, 0.0008, 0.37, 100000.6, 500.2 /)
  open (18, file='mydata.txt', status='replace')
  do i = 1, n
    write (18, *) x(i), y(i)
  end do
  close (18)
end program writeSimpleCSV
