#include "Python.h" //���python������

//void c_interface_python(float* state_1, float* reward, float* Done, int* Simu_Step_In, float* action_1, int* Simu_Step_Out)
void c_interface_python(int *Simu_Step_In, int *Simu_Step_Out)
{
	//Py_SetPythonHome(L"C:\\ProgramData\\Anaconda3\\envs\\py3.7\\python.exe");
	Py_Initialize(); //1����ʼ��python�ӿ�

	//��ʼ��ʹ�õı���
	PyObject *pModule = NULL;
	PyObject *pFunc = NULL;
	PyObject *pName = NULL;


	//2����ʼ��pythonϵͳ�ļ�·������֤���Է��ʵ� .py�ļ�
	PyRun_SimpleString("import sys");
	PyRun_SimpleString("sys.path.append('./')");

	//3������python�ļ�������ǰ�Ĳ���python�ļ�����ddpg_main.py����ʹ�����������ʱ��ֻ��Ҫд�ļ������ƾͿ����ˡ�����д��׺��
	pModule = PyImport_ImportModule("ddpg_main");

	if (!pModule) // ����ģ��ʧ��
	{
		printf("ERROR, Python get module failed.");
	}
	else
	{
		printf("INFO, Python get module succeed.\n");
	}
	//4�����ú���
	pFunc = PyObject_GetAttrString(pModule, "add");

	/*//5����python������
	PyObject* pArgs = PyTuple_New(4);//�������õĲ������ݾ�����Ԫ�����ʽ�����,2��ʾ�������������add��ֻ��һ������ʱ��д1�Ϳ����ˡ�����ֻ�Ƚ��ܺ��������в������ڵ������


	PyTuple_SetItem(pArgs, 0, Py_BuildValue("f", *state_1)); //0����ʾ��š���һ��������
	PyTuple_SetItem(pArgs, 1, Py_BuildValue("f", *reward)); //1��Ҳ��ʾ��š��ڶ���������i����ʾ����Ĳ���������int���͡�
	PyTuple_SetItem(pArgs, 2, Py_BuildValue("f", *Done)); //0����ʾ��š���һ��������
	PyTuple_SetItem(pArgs, 3, Py_BuildValue("i", *Simu_Step_In)); //1��Ҳ��ʾ��š��ڶ���������i����ʾ����Ĳ���������int���͡�*/

	//5����python������
	PyObject *pArgs = PyTuple_New(2);//�������õĲ������ݾ�����Ԫ�����ʽ�����,2��ʾ�������������add��ֻ��һ������ʱ��д1�Ϳ����ˡ�����ֻ�Ƚ��ܺ��������в������ڵ������
	float a = 3.1;
	float b = 4.2;
	PyTuple_SetItem(pArgs, 0, Py_BuildValue("f", a)); //0����ʾ��š���һ��������
	PyTuple_SetItem(pArgs, 1, Py_BuildValue("f", b)); //1��Ҳ��ʾ��š��ڶ���������i����ʾ����Ĳ���������int���͡�*/

	//6��ʹ��C++��python�ӿڵ��øú���
	PyObject *pReturn = PyEval_CallObject(pFunc, pArgs);
	PyObject* a1 = PyTuple_GetItem(pReturn, 0);
	PyObject* a2 = PyTuple_GetItem(pReturn, 1);
	float nResult1 = 0;
	float nResult2 = 0;
	PyArg_Parse(a1, "f", &nResult1);
	PyArg_Parse(a2, "f", &nResult2);
	//int nResult2 = 0;
	//PyObject *ob1, *ob2 = PyTuple_Unpack(pReturn);
	//7������python����õķ���ֵ
	//PyTuple_Unpack(pReturn);
	//PyFloat_AsDouble(pReturn);
	//PyArg_ParseTuple(pReturn, "i|i:ref", ob1, ob2);//i��ʾת����int�ͱ��������������Ҫע����ǣ�PyArg_Parse�����һ��������������ϡ�&�����š�
	//PyArg_Parse(pReturn, "f", &nResult1);//i��ʾת����int�ͱ��������������Ҫע����ǣ�PyArg_Parse�����һ��������������ϡ�&�����š�
	printf("The result is %f\n", nResult1);
	
	//8������python�ӿڳ�ʼ��
	Py_Finalize();
	*Simu_Step_Out = *Simu_Step_In + 1;
}