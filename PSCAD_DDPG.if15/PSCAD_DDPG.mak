
#------------------------------------------------------------------------------
# Project 'PSCAD_DDPG' make using the 'Intel(R) Visual Fortran Compiler 19.1.2.254 (64-bit)' compiler.
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# All project
#------------------------------------------------------------------------------

all: targets
	@echo !--Make: succeeded.



#------------------------------------------------------------------------------
# Directories, Platform, and Version
#------------------------------------------------------------------------------

Arch        = windows
EmtdcDir    = C:\PROGRA~2\PSCAD46\emtdc\if15
EmtdcInc    = $(EmtdcDir)\inc
EmtdcBin    = $(EmtdcDir)\$(Arch)
EmtdcMain   = $(EmtdcBin)\main.obj
EmtdcLib    = $(EmtdcBin)\emtdc.lib


#------------------------------------------------------------------------------
# Fortran Compiler
#------------------------------------------------------------------------------

FC_Name         = ifort.exe
FC_Suffix       = obj
FC_Args         = /nologo /c /free /real_size:64 /fpconstant /warn:declarations /iface:default /align:dcommons /fpe:0
FC_Debug        =  /O1
FC_Preprocess   = 
FC_Preproswitch = 
FC_Warn         = 
FC_Checks       = 
FC_Includes     = /include:"$(EmtdcInc)" /include:"$(EmtdcDir)" /include:"$(EmtdcBin)"
FC_Compile      = $(FC_Name) $(FC_Args) $(FC_Includes) $(FC_Debug) $(FC_Warn) $(FC_Checks)

#------------------------------------------------------------------------------
# C Compiler
#------------------------------------------------------------------------------

CC_Name     = cl.exe
CC_Suffix   = obj
CC_Args     = /nologo /MT /W3 /EHsc /c
CC_Debug    =  /O2
CC_Includes = 
CC_Compile  = $(CC_Name) $(CC_Args) $(CC_Includes) $(CC_Debug)

#------------------------------------------------------------------------------
# Linker
#------------------------------------------------------------------------------

Link_Name   = link.exe
Link_Debug  = 
Link_Args   = /out:$@ /nologo /nodefaultlib:libc.lib /nodefaultlib:libcmtd.lib /subsystem:console
Link        = $(Link_Name) $(Link_Args) $(Link_Debug)

#------------------------------------------------------------------------------
# Build rules for generated files
#------------------------------------------------------------------------------


.f.$(FC_Suffix):
	@echo !--Compile: $<
	$(FC_Compile) $<



.c.$(CC_Suffix):
	@echo !--Compile: $<
	$(CC_Compile) $<



#------------------------------------------------------------------------------
# Build rules for file references
#------------------------------------------------------------------------------


user_source_1.$(FC_Suffix): D:\C_PRAC~1\Pyhton_C\PYTHON~1\x64\Debug\FORTRA~2.F90
	@echo !--Compile: "D:\C_Practice\Pyhton_C\Python_Invoker\x64\Debug\fortran_interface_c_1.f90"
	copy "D:\C_Practice\Pyhton_C\Python_Invoker\x64\Debug\fortran_interface_c_1.f90" .
	$(FC_Compile) "fortran_interface_c_1.f90"
	del "fortran_interface_c_1.f90"

user_source_2.$(CC_Suffix): D:\C_PRAC~1\Pyhton_C\PYTHON~1\x64\Debug\C_INTE~1.C
	@echo !--Compile: "D:\C_Practice\Pyhton_C\Python_Invoker\x64\Debug\c_interface_python.c"
	copy "D:\C_Practice\Pyhton_C\Python_Invoker\x64\Debug\c_interface_python.c" .
	$(CC_Compile) "c_interface_python.c"
	del "c_interface_python.c"

#------------------------------------------------------------------------------
# Dependencies
#------------------------------------------------------------------------------


FC_Objects = \
 Station.$(FC_Suffix) \
 Main.$(FC_Suffix) \
 user_source_1.$(FC_Suffix)

FC_ObjectsLong = \
 "Station.$(FC_Suffix)" \
 "Main.$(FC_Suffix)" \
 "fortran_interface_c_1.$(FC_Suffix)"

CC_Objects = \
  user_source_2.$(CC_Suffix)

CC_ObjectsLong = \
  "c_interface_python.$(CC_Suffix)"

UserLibs =

SysLibs  = ws2_32.lib

Binary   = PSCAD_DDPG.exe

$(Binary): $(FC_Objects) $(CC_Objects) $(UserLibs)
	@echo !--Link: $@
	$(Link) "$(EmtdcMain)" $(FC_ObjectsLong) $(CC_ObjectsLong) $(UserLibs) "$(EmtdcLib)" $(SysLibs)

targets: $(Binary)


clean:
	-del EMTDC_V*
	-del *.obj
	-del *.o
	-del *.exe
	@echo !--Make clean: succeeded.



