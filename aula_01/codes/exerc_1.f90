program uniforme_estatisticas
    implicit none
    integer, parameter :: n = 100000
    integer :: i
    real :: u, soma, soma_q, media, variancia
    
    soma = 0.0
    soma_q = 0.0
    
    call random_seed()
    
    do i = 1, n
        call random_number(u)
        soma = soma + u
        soma_q = soma_q + (u * u)
    end do
    
    media = soma / real(n)
    ! Variancia amostral: E[X^2] - (E[X])^2
    variancia = (soma_q / real(n)) - (media * media)
    
    print *, 'Número de amostras: ', n
    print '(A, F10.6)', 'Média amostral:    ', media
    print '(A, F10.6)', 'Variância amostral: ', variancia
    print '(A, F10.6)', 'Valor exato esper.: ', 1.0 / 12.0
end program uniforme_estatisticas