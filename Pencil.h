#pragma once

#include <cstdlib>
#include <iostream>
#include<tuple>
#include <cassert>
#include<vector>
#include <functional>
#include <numeric>
#include <functional> 
#include <cstdlib>
#include<vector>
#include <execution>
#include <string>
#include "mpi.h"
#include "fftw3.h"


using namespace std;
#define SEQ std::execution::seq
#define pcout if (rank==0) cout

/**
 * @brief  block communication split for MPI
 * @param N				   global dimension
 * @param size			   MPI size
 * @param rank			   MPI rank
 */
inline tuple<int, int> blockdist(int N, int size, int i)
{
	auto [q, r] = std::div(N, size);
	// cout<<"q,r,N,size : "<<q<<" "<<r<<" "<<N<<" "<<size<<endl;
	int n  = q + (r > i ? 1 : 0);
	int s  = i * q + std::min(i,r);
	std::tuple<int, int> dist{n, s};
	// cout<<"n,s : "<<n<<" "<<s<<endl;
	return dist;
};

/**
 * @brief	 block communication subarrau for MPI_Type
 * @param comm			global communicator
 * @param shape		    global array shape
 * @param axis			distributed axes
 * @param subshape	    local pencil subshape
 * @param dtype		    MPI_Datatype
 */
inline vector<MPI_Datatype> subarraytypes(MPI_Comm comm, vector<int> shape, int axis, vector<int> subshape,MPI_Datatype dtype)
{
	int size = 0, rank = 0;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);

	auto subsizes = subshape;
	vector<int> substarts(subshape.size(), 0);
	vector<MPI_Datatype> datatypes{};
	auto N = shape[axis];

	for (int i = 0; i < size; ++i)
	{
		auto [n, s] = blockdist(N, size, i);
		subsizes[axis] = n;
		substarts[axis] = s;
		MPI_Datatype newtype{};
		// cout<<"N, size, i : "<<N<<" "<<size<<" "<<i<<endl;
		cout<<"here subtypes: shape, axis, subshape sizes, subsizes, substarts"<<":"
		<<shape[0]<<" "<<shape[1]<<" @ "<<axis<<" ; "<<subshape[0]<<" "<<subshape[1]<<" ; "<<subsizes[0]<<" "<<subsizes[1]<<substarts[0]<<" "<<substarts[1]<<endl;
		MPI_Type_create_subarray(int(shape.size()), shape.data(), subsizes.data(), substarts.data(),MPI_ORDER_C, dtype, &newtype);
		MPI_Type_commit(&newtype);
		datatypes.push_back(newtype);
	}
	return datatypes;
}

/**
 * @brief subarray communicator by MPI_Cart and subarray
 * @param comm 			global communicator
 * @param dims			procs of each dimensions
 * @param reorder		reorder of mpi_rank
 * @property			subcomms
 * @method				Subcomm
 */
struct Subcomm
{
	vector<MPI_Comm> subcomms;

	Subcomm(MPI_Comm comm, vector<int> dims, bool reorder = true)
	{
		int size = 0, rank = 0;
		MPI_Comm_size(comm, &size);
		MPI_Comm_rank(comm, &rank);
		MPI_Comm cart_comm;
		int ndims = static_cast<int>(dims.size());
		vector<int> periods(ndims, 0);
		MPI_Dims_create(size, ndims, dims.data());
		MPI_Cart_create(comm, ndims, dims.data(), periods.data(), reorder, &cart_comm);
		MPI_Cartdim_get(cart_comm, &ndims);

		MPI_Comm subtmp;
		auto remdims = new int[ndims]{0};
		for (int i = 0; i < ndims; ++i)
		{
			remdims[i] = 1;
			MPI_Cart_sub(cart_comm, remdims, &subtmp);
			subcomms.push_back(subtmp);
			remdims[i] = 0;
		}
		delete[] remdims;
		MPI_Comm_free(&cart_comm);
	}

    destroy() 
    {
        for(auto i:subcomms) MPI_Comm_free(&i);
    }
};

/**
 * @brief subarray communicator by Data Transpose
 * @param comm 			global communicator
 * @param dtype         MPI_Datatype
 * @param subshapeAB    pencil subshape
 * @param subtypesAB    subarraytypes
 * @property			
 * @method					Transfer  , forward/backword
 */
struct Transfer
{
	MPI_Comm comm;
	MPI_Datatype dtype;

	vector<int> shape;
	vector<int> subshapeA;
	vector<int> subshapeB;
	vector<MPI_Datatype> subtypesA;
	vector<MPI_Datatype> subtypesB;

	int axisA;
	int axisB;

	Transfer(MPI_Comm m_comm, vector<int> m_shape, MPI_Datatype m_dtype, vector<int> m_subA, int m_axisA,vector<int> m_subB, int m_axisB)
		: comm(m_comm), dtype(m_dtype), shape(m_shape), subshapeA(m_subA), subshapeB(m_subB), axisA(m_axisA),axisB(m_axisB)
	{

		subtypesA = subarraytypes(comm, shape, axisA, subshapeA, dtype);
		subtypesB = subarraytypes(comm, shape, axisB, subshapeB, dtype);
	}

	Transfer& operator =(Transfer& obj)
	{
		this->comm = obj.comm;
		this->dtype = obj.dtype;
		this->shape = obj.shape;

		this->subshapeA = obj.subshapeA;
		this->subshapeB = obj.subshapeB;
		this->axisA = obj.axisA;
		this->axisB = obj.axisB;

		return *this;
	}

	template <typename T>
	int forward(T* arrayA, T* arrayB , string direction="F")
	{
		int size = 0; 
		MPI_Comm_size(comm,&size);
		vector<int> counts {vector<int>(size,1)};
        vector<int> displs {vector<int>(size,0)};
		int status = 0;
		if (direction == "F")
		{
			status = MPI_Alltoallw(arrayA, counts.data(), displs.data(), subtypesA.data(),
		              			   arrayB, counts.data(), displs.data(), subtypesB.data(), comm);
		}
		else if (direction == "B")
		{
			status = MPI_Alltoallw(arrayB,  counts.data(), displs.data(), subtypesB.data(), 
					  			   arrayA,  counts.data(), displs.data(), subtypesA.data(), comm);
		}
		return status;
	}

    destroy()
    {
        for(auto i:subtypesA) MPI_Type_free(&i);
        for(auto i:subtypesB) MPI_Type_free(&i);
    }
};

/**
 *@brief    Pencil and slab decomposition 
 *@param comm       communicator
 *@param shape      global array dimension
 *@param axis       alignment dimension
 */
class Pencil
{
public:
	int axis;
	vector<int> shape;
	vector<MPI_Comm> subcomm;
	vector<int> subshape;
	vector<int> substarts;


	Pencil(vector<MPI_Comm> m_subcomm, vector<int> m_shape, int m_axis = -1)
		: shape(m_shape), axis(m_axis), subcomm(m_subcomm)
	{
		if (axis < 0) axis += static_cast<int>(shape.size());

		if (subcomm.size() < shape.size())
		{
			while (subcomm.size() < shape.size() - 1)
			{
				subcomm.push_back(MPI_COMM_SELF);
			}
			subcomm.insert(subcomm.begin() + axis, MPI_COMM_SELF);
		}

		for (int i = 0; i < subcomm.size(); ++i)
		{
			auto tmp = subcomm[i];
			int rank = 0, size = 0;
			MPI_Comm_size(tmp, &size);
			MPI_Comm_rank(tmp, &rank);
			auto [n, s] = blockdist(shape[i], size, rank);
			substarts.push_back(s);
			subshape.push_back(n);
		}
	}

	Pencil& operator =(Pencil& obj)
	{
		this->shape = obj.shape;
		this->axis = obj.axis;
		this->subcomm = obj.subcomm;
		this->substarts = obj.substarts;
		this->subshape = obj.subshape;
		return *this;
	}

    //axis_in is aligned dimension
	Pencil repencil(int axis_in)
	{
		if (axis_in < 0) axis_in += static_cast<int>(this->shape.size());
		// int i = this->axis;
		// int j = axis_in;

		// MPI_Comm tmp = subcomm[i];
		// subcomm[i] = subcomm[j];
		// subcomm[j] = tmp;
		auto subcomm_copy = subcomm;
        swap(subcomm_copy[this->axis], subcomm_copy[axis_in]);
		Pencil ptr = Pencil(subcomm_copy, this->shape, axis_in);
		return ptr;
	}
    //self->pencil   transfers
	Transfer transfer(Pencil pencil, MPI_Datatype dtype)
	{

		MPI_Comm comm_local             = this->subcomm[pencil.axis];
		vector<int> shape_local  = this->subshape;
		shape_local[pencil.axis]       = this->shape[pencil.axis];
		
		cout<<"transfer info : "<<shape_local[0]<<" "<<shape_local[1]<<"--"
		<<" " <<this->subshape[0]<<" " <<this->subshape[1]<<" " <<this->axis<<"--"
		<<" " <<pencil.subshape[0]<<" " <<pencil.subshape[1]<<" " <<pencil.axis<<" ";
		for (auto i:this->subcomm) {int tmp=0;MPI_Comm_size(i,&tmp);cout<<tmp;}
		cout<<endl;
		Transfer ptr = Transfer(comm_local, shape_local, dtype, this->subshape, this->axis, pencil.subshape, pencil.axis);
		return ptr;
	}
};
