program ridge_regression
 implicit none

 real(8), allocatable :: x(:), y(:), data(:), s(:), v(:)
 real(8), allocatable :: c(:, :), b(:, :), d(:, :), w(:)
 real(8) :: lambda, p, q, width, delta, xmin, z0, z1, x0, x1
 integer :: i, j, n, m, k, l, i1, i2

!****入力***************************************************
  ! データの数
  n = 50

 ! 近似したい曲線の次数
  m = 10

! ハイパーパラメータ
 lambda = 10.0_8 ** (0_8)

 !****データの読み込み***************************************************
allocate(x(n), y(n), data(2 * n))

  ! n×2のデータ（「data.txt」など）をサイズ2nの配列data(2*n)として読み込む（「./a.out < data.txt」 ）。
  ! data(2*n)はx,y,x,y,…のような並びになっている。
  read(*,*) data

  ! data(2*n)の奇数番目をx(j), 偶数番目をy(j)とする。
  do i = 1, 2*n
    if ( mod(i, 2) == 0 ) then
      j = i / 2
      y(j) = data(i)
    else if ( mod(i, 2) == 1 ) then
      j = (i + 1) / 2
      x(j) = data(i)
    end if
  end do

  ! 確認済み
  ! do j = 1, n
  !  write(*,*) j, x(j), y(j)
  ! end do

 !***係数などの計算*************************************************
 ! 解きたい連立方程式Cw=vのCとvの成分を計算しておく。
 ! 拡大係数行列D=[C v]を構築する。
  allocate(s(2*m + 1), v(m+1), b(m+1, m+1), c(m+1, m+1), d(m + 1, m + 2))

 ! s_k = sum_{i=1}^{n} {x_i}^k, k=0~2mの計算
  do k = 0, 2*m
    s(k) = 0.0_8
    do i = 1, n
      s(k) = s(k) + (x(i))**(real(k))
    end do
!    write(*,*) k, s(k) ! 確認済み
  end do

! 行列Bの計算。その成分はb_{ij}=s_{i+j-2}, i,j=1~m+1.
  do j = 1, m + 1
    do i = 1, m + 1
      b(i, j) = s(i + j - 2)
!      write(*,*) i, j, i + j - 2, b(i, j)  ! 確認済み
    end do
  end do

! 行列Cの計算。行列Bの対角成分にlambdaを足すだけ。
  do j = 1, m + 1
    do i = 1, m + 1
      if ( i == j ) then
        c(i, j) = b(i, j) + lambda
      else if ( i /= j ) then
        c(i, j) = b(i, j)
      end if
! 確認済み。データを最大(m+1)乗したものの総和をとるので、
! データが10^0のオーダーであっても、m=70などとするとc(m+1,m+1)は10^100くらいのオーダーになる
! ことがわかった。すなわちlambdaを10^1としてもc(i,i)-b(i,i)=0となってしまうので、
! lambdaはかなり大きい数でないといけないことがわかった。
!      write(*,*) i, j, c(i, j) - b(i, j)
    end do
  end do

 ! v_l = sum_{i=1}^{n} y_i {x_i}^l, l=0~mの計算
   do l = 0, m
     v(l) = 0.0_8
     do i = 1, n
       v(l) = v(l) + y(i) * (x(i))**l
     end do
!     write(*,*) l, v(l) ! 確認済み
   end do

 ! 拡大係数行列D=[C v]を作る
  do i = 1, m + 1
    d(i, m + 2) = v(i - 1)
  end do
  do j = 1, m + 1
    do i = 1, m + 1
      d(i, j) = c(i, j)
    end do
  end do
!   確認済み
!   do i = 1, m + 1
!     write(*,*) 'i-1=', i-1, 'v(i-1)=', v(i-1)
!     do j = 1, m + 2
!       write(*,*) i, j, d(i, j)
!     end do
!   end do

! ***ガウスの掃き出し法************************************************
! 上で作った(m+1)×(m+2)の拡大係数行列Dにガウスの掃き出し法を適用する。
! あるk行目の対角成分d(k,k)をpとおき、その行すべてをpで割る。
! そうすればその対角成分d(k,k)は1になる。
! k行目以外の行について（これをi行目とする）、d(k,k)と同じ列にあるもの(d(i,k))をqとおく。
! そのi行目のk列目以降の成分d(i,j), j=k~m+2 からq*d(k,j)を引く。
! これでd(k,k)の上下はゼロになる。
! このことをすべての行 k=1~m+1 について行う。

  do k = 1, m + 1
    p = d(k, k)
    do j = k, m + 2
      d(k, j) = d(k, j) / p
    end do
    do i = 1, m + 1
      if ( i /= k) then
        q = d(i, k)
        do j = k, m + 2
          d(i, j) = d(i, j) - q * d(k, j)
        end do
      end if
    end do
  end do

!   確認済み
!   do i = 1, m + 1
!     do j = 1, m + 2
!       write(*,*) i, j, d(i, j)
!     end do
!   end do

! こうして得られた行列の最後の列d(i,m+2), i=1~m+1がwの解である。
  allocate(w(m + 1))
  do i = 1, m + 1
    w(i - 1) = d(i, m + 2)
!    write(*,*) i - 1, w(i - 1) ! 確認済み
  end do

! ***曲線のプロット**********************************************************
! 分割数
  i1 = 200

! xの幅widthとxの最小値xminを計算。
  width = maxval(x) - minval(x)
  xmin = minval(x)
!    write(*,*) width, xmin, maxval(x) ! 確認済み

  ! 曲線のプロットの刻み幅
  delta = width / real( i1 )

  open (18, file='line.txt', status='replace')
  do i2 = 0, i1
    z1 = 0.0_8
    x1 = xmin + real(i2) * delta
      do i = 1, m + 1
        z1 = z1 + w(i) * x1**(real(i - 1))
      end do
    write(18, *) x1, z1 ! ここでxに対する近似曲線z=sum_{k=0}^{m} w_k x^kの値を出力
  end do
  close(18)

  deallocate(x, y, data, s, v, b, c, d, w)
 stop

end program
