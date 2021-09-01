!***********************************************************
! 最小二乗法
!***********************************************************
!
!***********************************************************
! 定数モジュール
!***********************************************************
module constants
  implicit none  ! 暗黙の型指定を使用しない
  ! データ数
  integer :: N
  ! 予測曲線の次数
  integer, parameter :: M = 5
  ! 測定データ
!  integer, parameter :: X(N) = (/ 1, 2, 3, 4, 5, 7, 7, 8 /)
!  integer, parameter :: Y(N) = (/ 8, 6, 5, 5, 3, 4, 6, 7 /)

  N=8
  real(8) :: x(N),y(N)

  x(N) = (/ 1.0_8, 2.0_8, 3.0_8, 4.0_8, 5.0_8, 7.0_8, 7.0_8, 8.0_8 /)
  y(N) = (/ 8.0_8, 6.0_8, 5.0_8, 5.0_8, 3.0_8, 4.0_8, 6.0_8, 7.0_8 /)

end module constants

!***********************************************************
! 計算モジュール
!***********************************************************
module calc
  use constants       ! 定数モジュール
  implicit none       ! 暗黙の型指定を使用しない
  private             ! モジュール外非公開
  public :: calc_lsm  ! calc_lsm のみモジュール外公開

contains

  !*********************************************************
  ! 最小二乗法
  !*********************************************************
  subroutine calc_lsm()
    ! 配列定義
    real :: a(M + 2, M + 1)
    real :: s(2 * M + 1) = 0, t(M + 1) = 0

    ! s[], t[] 計算
    call calc_st(s, t)

    ! a[][] に s[], t[] 代入
    call ins_st(a, s, t)

    ! 掃き出し
    call sweap_out(a)

    ! y 値計算＆結果出力
    call display(a, s, t)
  end subroutine calc_lsm

  !*********************************************************
  ! 以下、 private subroutine
  !*********************************************************
  ! s[], t[] 計算
  subroutine calc_st(s, t)
    real, intent(inout) :: s(:), t(:)
    integer :: i, j


    do i = 1, N
      do j = 1, 2 * M + 1
        s(j) = s(j) + x(i) ** (real(j - 1))
      end do
      do j = 1, M + 1
        t(j) = t(j) + x(i) ** real(j - 1) * y(i)
      end do
    end do
  end subroutine calc_st

  ! a[][] に s[], t[] 代入
  subroutine ins_st(a, s, t)
    real, intent(inout) :: a(:,:)
    real, intent(in)    :: s(:), t(:)
    integer :: i, j

    do i = 1, M + 1
      do j = 1, M + 1
        a(j, i) = s(i + j - 1)
      end do
      a(M + 2, i) = t(i)
    end do
  end subroutine ins_st

  ! 掃き出し
  subroutine sweap_out(a)
    real, intent(inout) :: a(:,:)
    integer :: i, j, k
    real    :: p, d

    do k = 1, M + 1
      p = a(k, k)
      do j = k, M + 2
        a(j, k) = a(j, k) / p
      end do
      do i = 1, M + 1
        if (i /= k) then
          d = a(k, i)
          do j = k, M + 2
            a(j, i) = a(j, i) - d * a(j, k)
          end do
        end if
      end do
    end do
  end subroutine sweap_out

  ! 結果出力
  subroutine display(a, s, t)
    real, intent(in) :: a(:,:)
    real, intent(in) :: s(:)
    real, intent(in) :: t(:)
    integer :: i, px
    real(8) :: p

    real(8) :: delta,x1,x0,f1,f0
    integer :: ne,imax
    ! neはnombre entiere(整数)

    ! 曲線のプロットの刻み幅
    delta=0.01_8
    ! 曲線のプロットの開始位置
    x0=-3.0_8
    !
    imax=int(6.0_8 / delta)
!             ↑　ここに測定データのXの幅をいれてください
!    write(*,*) delta,imax


!     do ne = 0,imax
!      f1 = 0.0_8
!      x1 = x0 + real(ne) * delta
!        do i = 1, M+1
!         f0 = a(M+2,i) * x1**(real(i-1))
!         f1 = f1 + f0
!        end do
!      write(*,*) x1,f1
!     end do

    do i = 1, M +1
      write (*, '(a,i1,a,f10.6)') "a", i - 1, " = ", a(M + 2, i)
    end do
    write (*, '(a)') "    x    y"
    do px = -3 * 2, 3 * 2
      p = 0.0_8
      do i = 1, M + 1
        p = p + a(M + 2, i) * ((px / 2.0) ** (i - 1))
      end do
      write (*, '(f5.1,f5.1)') (px / 2.0), p
    end do
  end subroutine display
end module calc

!***********************************************************
! 主処理
!***********************************************************
program least_squares_method
  use constants  ! 定数モジュール
  use calc       ! 計算モジュール
  implicit none  ! 暗黙の型指定を使用しない

  ! ==== 最小二乗法
  call calc_lsm()
end program least_squares_method
