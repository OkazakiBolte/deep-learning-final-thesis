program data_generator
 implicit none

 integer :: m, imax, i
 real(8) :: xmin, xmax, width, delta, x, y
 real(8), allocatable :: random(:)


! データの個数
  m = 30

  allocate(random(m + 1))
  call random_number(random)
  ! 確認
!  do i = 0, m
!   write(*,*) i, 2.5_8 * random(i + 1)
!  end do

! データの最小値・最大値の設定、幅、分割幅
  xmin = 0.0_8
  xmax = 6.0_8
  width = xmax - xmin
  delta = width / real(m)
!  write(*,*) width

  open(18, file='generated_data.txt', status='replace')
  do i = 0, m - 1
    x = xmin + delta * real(i)
    y = 2.0_8 * cos(2.0_8 * x) + x  + 2.0_8 * random(i + 1)
    write(18,*) x, y
  end do
  close(18)

  deallocate(random)
  stop
 end program
