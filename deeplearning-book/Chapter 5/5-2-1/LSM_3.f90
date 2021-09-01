!***********************************************************
! 定数モジュール
!***********************************************************
 module constants
   implicit none  ! 暗黙の型指定を使用しない
   ! データ数
   integer, parameter :: N = 8
   ! 予測曲線の次数
   integer, parameter :: M = 5
   ! 測定データ
   real(8) :: x(N),y(N)

   read(*,*) x
   read(*,*) y
   write(*,*) x,y
 end module constants



!***********************************************************
! 主処理
!***********************************************************
program LSM_3
 use constants


 stop
end program LSM_3
