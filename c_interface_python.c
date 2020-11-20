#include "Python.h" //添加python的声明

//void c_interface_python(float* state_1, float* reward, float* Done, int* Simu_Step_In, float* action_1, int* Simu_Step_Out)
void c_interface_python(int *Simu_Step_In, int *Simu_Step_Out)
{
	//Py_SetPythonHome(L"C:\\ProgramData\\Anaconda3\\envs\\py3.7\\python.exe");
	Py_Initialize(); //1、初始化python接口

	//初始化使用的变量
	PyObject *pModule = NULL;
	PyObject *pFunc = NULL;
	PyObject *pName = NULL;


	//2、初始化python系统文件路径，保证可以访问到 .py文件
	PyRun_SimpleString("import sys");
	PyRun_SimpleString("sys.path.append('./')");

	//3、调用python文件名。当前的测试python文件名是ddpg_main.py。在使用这个函数的时候，只需要写文件的名称就可以了。不用写后缀。
	pModule = PyImport_ImportModule("ddpg_main");

	if (!pModule) // 加载模块失败
	{
		printf("ERROR, Python get module failed.");
	}
	else
	{
		printf("INFO, Python get module succeed.\n");
	}
	//4、调用函数
	pFunc = PyObject_GetAttrString(pModule, "add");

	/*//5、给python传参数
	PyObject* pArgs = PyTuple_New(4);//函数调用的参数传递均是以元组的形式打包的,2表示参数个数。如果add中只有一个参数时，写1就可以了。这里只先介绍函数必须有参数存在的情况。


	PyTuple_SetItem(pArgs, 0, Py_BuildValue("f", *state_1)); //0：表示序号。第一个参数。
	PyTuple_SetItem(pArgs, 1, Py_BuildValue("f", *reward)); //1：也表示序号。第二个参数。i：表示传入的参数类型是int类型。
	PyTuple_SetItem(pArgs, 2, Py_BuildValue("f", *Done)); //0：表示序号。第一个参数。
	PyTuple_SetItem(pArgs, 3, Py_BuildValue("i", *Simu_Step_In)); //1：也表示序号。第二个参数。i：表示传入的参数类型是int类型。*/

	//5、给python传参数
	PyObject *pArgs = PyTuple_New(2);//函数调用的参数传递均是以元组的形式打包的,2表示参数个数。如果add中只有一个参数时，写1就可以了。这里只先介绍函数必须有参数存在的情况。
	float a = 3.1;
	float b = 4.2;
	PyTuple_SetItem(pArgs, 0, Py_BuildValue("f", a)); //0：表示序号。第一个参数。
	PyTuple_SetItem(pArgs, 1, Py_BuildValue("f", b)); //1：也表示序号。第二个参数。i：表示传入的参数类型是int类型。*/

	//6、使用C++的python接口调用该函数
	PyObject *pReturn = PyEval_CallObject(pFunc, pArgs);
	PyObject* a1 = PyTuple_GetItem(pReturn, 0);
	PyObject* a2 = PyTuple_GetItem(pReturn, 1);
	float nResult1 = 0;
	float nResult2 = 0;
	PyArg_Parse(a1, "f", &nResult1);
	PyArg_Parse(a2, "f", &nResult2);
	//int nResult2 = 0;
	//PyObject *ob1, *ob2 = PyTuple_Unpack(pReturn);
	//7、接收python计算好的返回值
	//PyTuple_Unpack(pReturn);
	//PyFloat_AsDouble(pReturn);
	//PyArg_ParseTuple(pReturn, "i|i:ref", ob1, ob2);//i表示转换成int型变量。在这里，最需要注意的是：PyArg_Parse的最后一个参数，必须加上“&”符号。
	//PyArg_Parse(pReturn, "f", &nResult1);//i表示转换成int型变量。在这里，最需要注意的是：PyArg_Parse的最后一个参数，必须加上“&”符号。
	printf("The result is %f\n", nResult1);
	
	//8、结束python接口初始化
	Py_Finalize();
	*Simu_Step_Out = *Simu_Step_In + 1;
}