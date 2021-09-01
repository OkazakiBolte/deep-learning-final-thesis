program thisisapen
 implicit none

 character(len = 16) :: a, b, c, d
 integer :: x, y

  a = 'This'
  b = 'is'
  c = 'a'
  d = 'pen.'
  x = 1
  y = 2

  write(*,'(a,x,a,x,a,x,a)') trim(a), trim(b), trim(c), trim(d)
  write(*,*) x, y

 stop
end program
