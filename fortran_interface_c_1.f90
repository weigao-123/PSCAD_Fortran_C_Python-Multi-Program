subroutine fortran_interface_c_1(Simu_Step_In,Simu_Step_Out)
    implicit none
    integer :: Simu_Step_In, Simu_Step_Out

! Fortran 90 interface to a C procedure
        interface
            subroutine c_interface_python(Simu_Step_In,Simu_Step_Out)
            
                !DEC$ ATTRIBUTES C :: c_interface_python

                !DEC$ ATTRIBUTES REFERENCE :: Simu_Step_In
            
                !DEC$ ATTRIBUTES REFERENCE :: Simu_Step_Out

!

! in, out are passed by REFERENCE

!
            
                integer :: Simu_Step_In, Simu_Step_Out
                
            end subroutine c_interface_python
        end interface

!

! Call the C procedure

!

      call c_interface_python(Simu_Step_In,Simu_Step_Out)

end subroutine fortran_interface_c_1

 
