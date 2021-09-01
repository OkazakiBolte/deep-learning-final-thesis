program ridge_regression
 implicit none

 real(8), allocatable :: x(:), y(:), data(:), s(:), v(:)
 real(8), allocatable :: a(:, :), b(:, :), c(:, :), w(:)
 real(8) :: lambda, p, q, width, delta, xmin, y_hat, x1
 integer :: i, j, n, m, k, l, i1, i2

 ! ***入力*********************
   ! データの数
   m = 2

   ! 近似したい曲線の次数
   n = 3

   ! 正則化パラメータ
   lambda = 0.0_8

 !***データの読み込み***********************
  allocate(x(m), y(m), data(2 * m))
  ! m×2のデータ（「data.txt」など）をサイズ2mの配列data(2*m)として読み込む（「./a.out < data.txt」 ）。
  ! data(2*m)はx,y,x,y,…のような並びになっている。

  read(*,*) data

  ! data(2*m)の奇数番目をx(j), 偶数番目をy(j)とする。
  do i = 1, 2*m
    if ( mod(i, 2) == 0 ) then
      j = i / 2
      y(j) = data(i)
    else if ( mod(i, 2) == 1 ) then
      j = (i + 1) / 2
      x(j) = data(i)
    end if
  end do

! 確認済み
! do j = 1, m
!  write(*,*) j, x(j), y(j)
! end do

!***拡大係数行列の準備***********************************
! 解きたい連立方程式Bw=vのBとvの成分を計算しておく。
! 拡大係数行列C=[B v]を構築する。
  allocate(s(0:2 * n), a(n + 1, n + 1), b(n + 1, n + 1), v(0 : n), c(n + 1,  n + 2))

! s_k = sum_{i=1}^{m} {x_i}^k, k=0~2nの計算
  do k = 0, 2 * n
    s(k) = 0.0_8
    do i = 1, m
      s(k) = s(k) + (x(i)) ** (real(k))
    end do
!    write(*,*) k, s(k)
  end do

  ! 行列Aの計算。その成分はa_{ij}=s_{i+j-2}, i,j=1~n+1.
  do j = 1, n + 1
    do i = 1, n + 1
      a(i, j) = s(i + j - 2)
  !    write(*,*) i, j, i + j - 2, s(i + j - 2), a(i, j)  ! 確認済み
    end do
  end do

  ! 行列Bの計算。行列Aの対角成分にlambdaを足すだけ。
  do j = 1, n + 1
    do i = 1, n + 1
      if ( i == j ) then
        b(i, j) = a(i, j) + lambda
      else if ( i /= j ) then
        b(i, j) = a(i, j)
      end if
! 確認済み。データを最大(2n)乗したものの総和をとるので、
! データが10^0のオーダーであっても、n=30などとするとc(n+1,n+1)は10^60くらいのオーダーになる
! ことがわかった。すなわちlambdaを10^1としてもb(i,i)-a(i,i)=0となってしまうので、
! lambdaはかなり大きい数でないといけないことがわかった。あるいはnをちいさくする。
!      write(*,*) i, j, b(i, j) - a(i, j)
    end do
  end do

! v_l = sum_{i=1}^{m} y_i {x_i}^l, l=0~nの計算
   do l = 0, n
     v(l) = 0.0_8
     do i = 1, m
       v(l) = v(l) + y(i) * (x(i)) ** real(l)
     end do
!     write(*,*) l, v(l) ! 確認済み
   end do

! 拡大係数行列C=[B v]を作る
  do i = 0, n
    do j = 1, n + 2
      if (j == n + 2) then
        c(i + 1, j) = v(i)
      else if (j /= n + 2) then
        c(i + 1, j) = b(i + 1, j)
      end if
    end do
  end do
!   確認済み
  do i = 0, n
    do j = 1, n + 2
!      write(*,*) i + 1, j, v(i), c(i + 1, j)
    end do
  end do

! ***ガウスの掃き出し法**************************************
! 上で作った(n+1)×(n+2)の拡大係数行列Cにガウスの掃き出し法を適用する。
! あるk行目の対角成分c(k,k)をpとおき、その行すべてをpで割る。
! そうすればその対角成分c(k,k)は1になる。
! k行目以外の行について（これをi行目とする）、c(k,k)と同じ列にあるもの(c(i,k))をqとおく。
! そのi行目のk列目以降の成分c(i,j), j=k~n+2 からq*c(k,j)を引く。
! これでc(k,k)の上下はゼロになる。
! このことをすべての行 k=1~n+1 について行う。

  do k = 1, n + 1
    p = c(k, k)
    do j = k, n + 2
      c(k, j) = c(k, j) / p
    end do
    do i = 1, n + 1
      if (i /= k) then
        q = c(i, k)
        do j = k, n + 2
          c(i, j) = c(i, j) - q * c(k, j)
        end do
      end if
    end do
  end do

!   確認済み
!   do i = 1, n + 1
!     do j = 1, n + 2
!       write(*,*) i, j, c(i, j)
!     end do
!   end do

! こうして得られた行列の最後の列c(i,n+2), i=1~n+1がwの解である。
  allocate(w(n + 1))
    do i = 1, n + 1
      w(i) = c(i, n + 2)
      write(*,*)  i - 1, w(i) ! 確認済み
    end do

! ***曲線のプロット****************************************
! 分割数
  i1 = 200

! xの幅widthとxの最小値xminを計算。
  width = maxval(x) - minval(x)
  xmin = minval(x)
!    write(*,*) width, xmin, maxval(x) ! 確認済み

  ! 曲線のプロットの刻み幅
  delta = width / real( i1 )

  open (18, file='essayant.txt', status='replace')
  do i2 = 0, i1
    y_hat = 0.0_8
    x1 = xmin + real(i2) * delta
      do i = 1, n + 1
        y_hat = y_hat + w(i) * x1 ** (real(i - 1))
      end do
    write(18, *) x1, y_hat ! ここでxに対する近似曲線y_hat=sum_{k=0}^{n} w_k x^kの値を出力
  end do
  close(18)

  deallocate(x, y, data, s, v, a, b, c, w)

 stop
end program
