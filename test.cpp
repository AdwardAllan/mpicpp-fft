#include <cstdlib>
#include <iostream>
#include<tuple>
#include <cassert>
#include<vector>
#include<array>
#include "mpi.h"
#include "Pencil.h"




using    namespace std;

int      main(int argc,char** argv)
{

         MPI_Init(&argc, &argv);
         MPI_Comm comm     = MPI_COMM_WORLD;
         int rank          = 0;
         MPI_Comm_rank(comm, &rank);
   
         vector <int> N {4,4,4};
         vector <int> dims{2,2,1};
         Subcomm subcomm   = Subcomm(comm,dims);

         int tmp           = 0;
         for (auto i:subcomm.subcomms)
         {
         MPI_Comm_size(i, &tmp);
         pcout << tmp << endl;
         }

         Pencil p0         = Pencil(subcomm.subcomms, N, 2);

         Pencil p1         = p0.repencil(0);

         Transfer pp       = p0.transfer(p1, MPI_INT);


         for (auto i : p0.subshape) pcout << i <<" ";         pcout << endl;
         for (auto i : p1.subshape) pcout << i << " ";         pcout << endl;


         int arrayA[2][2][4] {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
         for (int i = 0; i < 2; ++i)
         {
            for (int j=0;j<2;++j)
            {
                for (int k=0;k<4;++k)
	            {arrayA[i][j][k] += rank;}
            }

         }
         int arrayB[4][4][4] {0};
        
		// int sendcounts[4]{1,1,1,1};
        // // int sendcounts_test[4]{1};
		// int recvcounts[4]{1,1,1,1};
		// int senddispls[4]{};
		// int recvdispls[4]{};

        vector<int> counts {vector<int>(4,1)};
        vector<int> displs {vector<int>(4,0)};
        // cout<<"sendcounts_test"<<sendcounts_test[2]<<endl;
		// MPI_Alltoallw(&(arrayA[0][0]), counts.data(), displs.data(), pp.subtypesA.data(), &(arrayB[0][0]),  counts.data(), displs.data(),pp.subtypesB.data(), pp.comm);
		cout << "forward" << endl;

         MPI_Barrier(comm);
         //
         pp.forward<int>(&(arrayA[0][0][0]), &(arrayB[0][0][0]));

         MPI_Barrier(comm);
         for (int i = 0; i < 2; ++i)
         {
            for (int j=0;j<2;++j)
            {
                for (int k=0;k<4;++k)
                {
                    cout << arrayA[i][j][k] << "   ";
                }
	            cout<< " & ";
            }
             cout<< " & ";
         }
         cout <<" <- A"<< endl;
         MPI_Barrier(comm);

         for (int i = 0; i < 4; ++i)
         {
             for (int j = 0; j < 2; ++j)
             {
                for (int k = 0; k < 2; ++k)
                {
                    cout << arrayB[i][j][k] <<"  ";
                }
                cout<< " & ";
             }
             cout<< " & ";
         }
         cout << " ->B" << endl;
         MPI_Barrier(comm);
        // pp.backward<int>(&(arrayB[0][0]), &(arrayA[0][0]));


         MPI_Finalize();
         return 0;
}