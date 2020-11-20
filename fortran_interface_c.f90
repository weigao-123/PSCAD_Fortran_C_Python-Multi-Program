subroutine fortran_interface_c(state_1,reward,Done,Simu_Step_In,action_1,Simu_Step_Out)
    implicit none
    real :: state_1, action_1
    real :: reward, Done
    integer :: Simu_Step_In, Simu_Step_Out

! Fortran 90 interface to a C procedure
        interface
            subroutine c_interface_python(state_1,reward,Done,Simu_Step_In,action_1,Simu_Step_Out)
            
                !DEC$ ATTRIBUTES C :: c_interface_python

                !DEC$ ATTRIBUTES REFERENCE :: state_1

                !DEC$ ATTRIBUTES REFERENCE :: reward
            
                !DEC$ ATTRIBUTES REFERENCE :: Done

                !DEC$ ATTRIBUTES REFERENCE :: Simu_Step_In
            
                !DEC$ ATTRIBUTES REFERENCE :: action_1

                !DEC$ ATTRIBUTES REFERENCE :: Simu_Step_Out

!

! in, out are passed by REFERENCE

!
                real :: state_1, action_1
                real :: reward, Done
                integer :: Simu_Step_In, Simu_Step_Out
                
            end subroutine c_interface_python
        end interface

!

! Call the C procedure

!

      call c_interface_python(state_1,reward,Done,Simu_Step_In,action_1,Simu_Step_Out)

end subroutine fortran_interface_c

 
