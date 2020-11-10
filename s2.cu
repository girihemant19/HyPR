#include <stdio.h>
#include <stdlib.h>
//#include <math.h>
//#include <string.h>
#include <chrono>
#include <omp.h>
#include <vector>
#include <stdio.h>
#include <iostream>
#include <set>
#include <iterator>
#include <algorithm>
#include <bits/stdc++.h>
#include "json/json/json.h"
#include "json/json/reader.h"
#include "json/json/value.h"
#include <cuda.h>

#define N 512
#define BLOCK_DIM 512
#define R 16//row or threads
#define C 5000// columns
#define ind(i,j,n) ((i*n) + j)
#define STR(int) to_string(int)

using std::mt19937_64;
using std::random_device;
using std::uniform_int_distribution;
using std::vector;
using std::cout;
using std::cerr;
using std::endl;
using std::out_of_range;


double epsilon;
#define GRAPH_FILE_SEPERATOR " ,;       "
#define MAX_LINE_LEN 100000


long string_to_long(char *str)
{
        long val;
        char *endptr;
        errno = 0;
    val = strtol(str, &endptr, 10);
        if((errno == ERANGE && (val == LONG_MAX || val == LONG_MIN)) || (errno != 0 && val == 0) || (endptr == str))
        {
                perror("Error while converting string to long value");
                val = -1;
        }
        return val;
}

__global__ void scaling_v_old(double *l_ranks,int *l_old ,int l_V, int l_graph,int l_size )
                {
                int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
                int i= blockId * (blockDim.x * blockDim.y * blockDim.z)+ (threadIdx.z * (blockDim.x * blockDim.y))+ (threadIdx.y * blockDim.x)+ threadIdx.x;
                float k=static_cast<double>(l_V)/static_cast<double>(l_graph);
                
                if(i<l_size)
                {
			//printf("%d ", i);
                        l_ranks[l_old[i]]=l_ranks[l_old[i]]*k;
                }      
                }
__global__ void scaling_v_border(double *l_ranks, int *l_border, int l_V, int l_vpagerank,int l_size )
                {
                int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
                int i= blockId * (blockDim.x * blockDim.y * blockDim.z)+ (threadIdx.z * (blockDim.x * blockDim.y))+ (threadIdx.y * blockDim.x)+ threadIdx.x;
                 //printf("%d ", i);  
                float k=static_cast<double>(l_V)/static_cast<double>(l_vpagerank);
                if(i<l_size)
                {
                        l_ranks[l_border[i]]=l_ranks[l_border[i]]*k;
                } 
                }
__global__ void ranks_sum_cal(double *dest0,int *l_temp,int *out_link, int *l_pred2, double *ranks3, int l_size,int l_sz)
                {
                int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
                int i= blockId * (blockDim.x * blockDim.y * blockDim.z)+ (threadIdx.z * (blockDim.x * blockDim.y))+ (threadIdx.y * blockDim.x)+ threadIdx.x; 
                double rank_sum;
                if(i<C*R)
                if(i<l_size )//&& l_temp[i]!=0 )
                {
                        for(int l=0;l<100;l++)
                        {
                                int element=l_pred2[l];
                                int outlnks=out_link[element];
                                if(outlnks==0)
                                {
                                        outlnks=1;
                                }
                                rank_sum+=(1/float(outlnks))*ranks3[element];  
                        }
                        dest0[i]=rank_sum;
                }
                }
__global__ void pagerank_aff(double *dest, double *l_ranks_sum_of_node,double l_d, int l_nvpg, int l_size)
                {
                int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
                int i= blockId * (blockDim.x * blockDim.y * blockDim.z)+ (threadIdx.z * (blockDim.x * blockDim.y))+ (threadIdx.y * blockDim.x)+ threadIdx.x;
                double k=(1-l_d)/(l_nvpg);
                if (i < l_size)
                {
                dest[i]= k+ (l_d*l_ranks_sum_of_node[i]);
                }
                }

using namespace std;
void euclidean_cal(double *, double*, double *, int);
int findElement(int arr[],int start, int end,int key);  
 
int deleteElement(int arr[], int n,  
    int key)  
{  
int pos = findElement(arr,0,n,key);  
int i;  
    for (i = pos; i < n - 1; i++)  
        arr[i] = arr[i + 1];  
  
    return n - 1;

return n - 1;  
}  

/*int findElement(int arr[], int key, int start, int end)  
{  
int i;  
for (i =start ; i <= end; i++)  
if (arr[i] == key)  
return i;  

return - 1;  
}  */
int findElement(int arr[], int l, int r, int x) 
{ 
if (r >= l) { 
int mid = l + (r - l) / 2; 

// If the element is present at the middle 
// itself 
if (arr[mid] == x) 
return mid; 

// If element is smaller than mid, then 
// it can only be present in left subarray 
if (arr[mid] > x) 
return findElement(arr, l, mid - 1, x); 

// Else the element can only be present 
// in right subarray 
return findElement(arr, mid + 1, r, x); 
} 

// We reach here when element is not 
// present in array 
return -1; 
} 
int deleteElement1(int arr[], int n, int x) 
{ 
// If x is last element, nothing to do 
if (arr[n-1] == x) 
return (n-1); 

// Start from rightmost element and keep moving 
// elements one position ahead. 
int prev = arr[n-1], i; 
for (i=n-2; i>=0 && arr[i]!=x; i--) 
{ 
int curr = arr[i]; 
arr[i] = prev; 
prev = curr; 
} 

// If element was not found 
if (i < 0) 
return 0; 

// Else move the next element in place of x 
arr[i] = prev; 

return (n-1); 
} 

class RandomIterator
{
    public:
        RandomIterator(const unsigned long long &amount, const unsigned long long &min, const unsigned long long &max): gen((random_device())())

        {
            floor = min;
            num_left = amount;
            last_k = min;
            n = max;
        }

        const bool has_next(void)
        {
            return num_left > 0;
        }

        const unsigned long long next(void)
        {
            if (num_left > 0)
            {
                unsigned long long range_size = (n - last_k) / num_left;
                
                uniform_int_distribution<unsigned long long> rnd(floor, range_size);

                unsigned long long r = rnd(gen) + last_k + 1;

                last_k = r;
                num_left--;
                return r;
            }
            else
            {
                throw out_of_range("Exceeded amount of random numbers to generate.");
            }
        }
    private:
            unsigned long long floor;
            unsigned long long n;
            unsigned long long last_k;
            unsigned long long num_left;
            mt19937_64         gen;

};

int largest(int arr[], int n) 
{ 
        return *max_element(arr, arr+n); 
} 
int main(int argc, char *argv[])
{
        double time_spent=0.0;
        std::ifstream is(argv[2]);
        std::istream_iterator<double> start(is), end;
        std::vector<double> ranks(start, end);
        for(int f=0;f<10;f++)
        { 
                FILE *file1,*file2;
                //char *output_file = "results_openmp.txt";
///////////////////////////////////////////////MEMORY STATUS///////////////////////////////////////////////////////////////
                cudaError_t stat = cudaSuccess;
                int devCount = 0;
                cudaGetDeviceCount( &devCount );
                
                if( !devCount )
                {
                        cout << "No Cuda Device." << endl;
                        return 0;
                }
                
                cudaDeviceProp devProp;
                cudaGetDeviceProperties( &devProp, 0 );
                cout << "Global Memory size: " << devProp.totalGlobalMem << " bytes" << endl;
                
                size_t free_bytes=0, total_bytes=0;
                cudaMemGetInfo( &free_bytes, &total_bytes );
                cout << "Free : " << free_bytes << " bytes" << endl;
                cout << "Total: " << total_bytes << " bytes" << endl;
                
                // cudaDeviceSetLimit has no effects!
                {
                        size_t sizeLimit = 0;
                        cudaDeviceGetLimit( &sizeLimit, cudaLimitMallocHeapSize );
                        cout << "sizeLimit: " << sizeLimit << endl;
                
                        cudaDeviceSetLimit( cudaLimitMallocHeapSize, free_bytes*1024*1024 );
                
                        cudaDeviceGetLimit( &sizeLimit, cudaLimitMallocHeapSize );
                        cout << "sizeLimit: " << sizeLimit << endl;
                }
                
                size_t* reservedMemory = (size_t*)NULL;
                stat = cudaMalloc( &reservedMemory, free_bytes+1 ); // one more bytes
                if( stat != cudaSuccess )
                {
                        cout << "Failed to allocate memory." << endl;
                }
                if( reservedMemory ) { cudaFree( reservedMemory ); }
                cudaDeviceReset();
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                double time_spent1=0.0;
                double time_spent2=0.0;
                cudaEvent_t tstart, tstop;
	        float elapsed_gpu;
                cout<<"\nBatch_"<<f;
                printf("\nPreProcessing Start\n");
                unsigned long long amount = atoi(argv[1]);//number of nodes to be deleted
                unsigned long long min = 0;
                unsigned long long max = ranks.size();
                set<int> del;
                if (amount > max)
                {
                        cerr << "Amount must be less than max." << endl;
                        return -1;
                }

                RandomIterator iterator(amount, min, max);
                while(iterator.has_next())
                {
                        del.insert(iterator.next());
                }
                char *token1, *token2;
                char line[MAX_LINE_LEN];

                long pnode, cnode;
                set <int, greater <int> > v_old;
                set <int, greater <int> > v_new;
                set <int, greater <int> > v_border;
                set <int, greater <int> > v_temp;
                set <int, greater <int> > result;
                set <int, greater <int> > result1;
                Json::Value root;
                Json::Value root1;

                omp_set_num_threads(R);
              
                string s1="batches/oldbatch_"+std::to_string(f)+".txt";
                string s2="batches/newbatch_"+std::to_string(f)+".txt";
                file1 = fopen(s1.c_str(),"r");//old_batch
                
                if(file1)
                {       
                        while (fgets(line, sizeof(line), file1))
                        {
                                token1 = strtok (line,GRAPH_FILE_SEPERATOR);
                                token2 = strtok(NULL,GRAPH_FILE_SEPERATOR);
                                pnode = string_to_long(token1);
                                cnode = string_to_long(token2);
                                if(pnode < 0 || cnode < 0)
                                        return -1;
                                //printf("%d %d\n",pnode,cnode);
                v_old.insert(pnode);
                v_old.insert(cnode);

                        }
                }
                else
                        return -1;
                
                std::set<int>::iterator iter = del.begin();
                while (iter != del.end())
                {
                        v_old.erase(*iter);
                        ranks[*iter]=0.0;
                        iter++;
                }

                file2 = fopen(s2.c_str(),"r");//new_batch
                if(file2)
                {       
                        while (fgets(line, sizeof(line), file2))
                        {
                                token1 = strtok (line,GRAPH_FILE_SEPERATOR);
                                token2 = strtok(NULL,GRAPH_FILE_SEPERATOR);
                                
                                pnode = string_to_long(token1);
                                cnode = string_to_long(token2);
                                if(pnode < 0 || cnode < 0)
                                        return -1;
                                //printf("%d %d\n",pnode,cnode);
                v_new.insert(pnode);
                v_new.insert(cnode);

                        }
                }
                else
                        return -1;


                std::set_difference(v_old.begin(),v_old.end(),v_new.begin(),v_new.end(),std::inserter(result, result.end()));
                
                v_old=result;
                
                std::set<int>::iterator it1 = v_new.begin();
                int *vnew;
                vnew = (int*)calloc(  v_new.size(), sizeof(int) );
                int a=0;
                while (it1 != v_new.end())
                {
                        vnew[a]=(*it1);
                        it1++;
                        a++;
                }
                //cout<<a<<endl;
                std::set<int>::iterator it = v_old.begin();
                int *vold;
                vold = (int*)calloc(  v_old.size(), sizeof(int) );
                int b=0;
                while (it != v_old.end())
                {
                        vold[b]=*it;
                        it++;
                        b++;
                }
                
                ifstream file("graph/Succ_"+std::to_string(f)+".txt");
                file >> root;

                ifstream file3("graph/Pred_"+std::to_string(f)+".txt");//predecessor
                file3 >> root1;
                int V=root.size();
                int V1=root1.size();
                
                int *vtemp1;
                vtemp1 = (int*)calloc(  R*C, sizeof(int) );
                

                int *vborder;
                vborder = (int*)calloc(  R*C, sizeof(int) );

                int *times;
                times = (int*)calloc(  R, sizeof(int) );
                //int x;
                int **succ1 = (int **)malloc(V *sizeof(int *)); 
                for (int i=0; i<V; i++) 
                        succ1[i] = (int *)malloc(100 * sizeof(int));
                
                for(int i=0;i<V;i++)
                        for(int j=0;j<100;j++)
                        {
                                //x++;
                                succ1[i][j]=root[to_string(i)][j].asInt();
                        }
                
                int **pred1 = (int **)malloc(V1 * sizeof(int *)); 
                for (int i=0; i<V1; i++) 
                        pred1[i] = (int *)malloc(100 * sizeof(int));
                
                for(int i=0;i<V1;i++)
                        for(int j=0;j<100;j++)
                        {
                                
                                pred1[i][j]=root1[to_string(i)][j].asInt();
                        }
                int z=0,z1=0;
                #pragma omp parallel //preprocessing start
		    {
			int tid=omp_get_thread_num();
			if(tid>0)
			{
				int st=a;
				int block_size = (st / (R-1)) + ((st % (R-1)) != 0);//ceil
				int start=(tid-1)*block_size;
				int end=(start+block_size)-1;

				int st2=C, y, i, n, x;
				int block_size1 = (st2 / (R-1)) + ((st2 % (R-1)) != 0);
				int start1=(tid-1)*block_size1;
				int end1=(start1+block_size1)-1;

				auto time1 = chrono::steady_clock::now();
				//#pragma omp barrier
				for(i=start;i<end;i++)
				{
                                        if(i<st)
					{
                                                n=vnew[i];
                                                vtemp1[ind(0,i,100)]=n;
                                                if(n<V){
                                                for(x=0;x<100;x++)//successor
                                                {
                                                        vtemp1[ind(tid,i+(x),100)] = succ1[n][x];
                                                        
                                                        
                                                }}
                                        }
				}
                                for(i=start1;i<end1;i++)
                                {
                                        if(i<st2)
                                        {  
                                                n=vtemp1[i];
                                                if(n<V1){
                                                for(x=0;x<100;x++)
                                                {
                                                        vborder[ind(tid,i+x,100)]=pred1[n][x];
                                                        
                                                }}
                                        }       
                                }
                                auto time2 = chrono::steady_clock::now();
                                times[tid]= chrono::duration_cast<chrono::milliseconds>(time2 - time1).count();
                	}
                }
                int maxi=largest(times, R);
                cout << "\nTime for Preprocessing : "<< maxi<< " ms" << endl;
                printf("\nPreProcessing Ends\n");
                root.clear();
                ifstream file4("graph/Outll_"+std::to_string(f)+".txt");
                file4 >> root;
                int *outll;
                outll = (int*)calloc(  R*C, sizeof(int) );
                for(int i=0;i<R*C;i++)
                {
                        if(vtemp1[i]!=0)
                        {
                        outll[i]=root[to_string(vtemp1[i])].asInt();
                        z++;
                        }
                }
                for(int i=0;i<C*R;i++)
                {
                        if(vborder[i]!=0)
                        {
                        outll[i]=root[to_string(vborder[i])].asInt();
                        z1++;
                        }
                }
                root.clear();
                ifstream file5("graph/Pred_"+std::to_string(f)+".txt");
                file5 >> root;

                
                int *pred2;
                pred2 = (int*)calloc(  root.size()*1000, sizeof(int) );

                for(int i=0;i<root.size();i++)
                for(int j=0;j<100;j++)
                {
                        pred2[ind(i,j,100)]=root[to_string(i)][j].asInt();
                        //cout<<pred2[ind(i,j,10)]<<i<<" i "<<j<<" j ";
                }   
                cout<<"New Nodes: "<<z<<endl<<"Border nodes: "<<z1<<endl<<"Old nodes: "<<v_old.size()<<endl;
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////preprocessing ends
                double *ranks1;
                ranks1=(double*)calloc( ranks.size() , sizeof(double) );

                std::copy(ranks.begin(), ranks.end(), ranks1);
                double *l_ranks;
                int *l_old;//*dest;
                int *l_border;
                int sz=v_old.size();
		printf("%d ", sz);
                int sz1=z1;//v_border.size();
                int V2=z;//v_temp.size();
                double trans_elapsed=0.0;
                
                cudaMalloc(&l_old, v_old.size()*sizeof(int));
                cudaMalloc(&l_border, sz1*sizeof(int));
                cudaMalloc(&l_ranks, ranks.size()*sizeof(double));
                auto trans1 = chrono::steady_clock::now();
                if(cudaMemcpy(l_ranks, ranks1, ranks.size()*sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess)
                {
                        printf("Error 0 in cudaMemcpy\n");
                        return -2;
                }
                if(cudaMemcpy(l_old, vold, v_old.size()*sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
                {
                        printf("Error 1 in cudaMemcpy\n");
                        return -2;
                }
                if(cudaMemcpy(l_border, vborder, sz1*sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
                {
                        printf("Error 2 in cudaMemcpy\n");
                        return -2;
                }
                auto trans2 = chrono::steady_clock::now();
                trans_elapsed= chrono::duration_cast<chrono::milliseconds>(trans2 - trans1).count();
                //cout<<trans_elapsed<<endl;
                time_spent2+=trans_elapsed;
                cudaEventCreate(&tstart);
                cudaEventCreate(&tstop);
                cudaEventRecord(tstart, 0);
                scaling_v_old<<<1024,1024,1024>>>(l_ranks,l_old, V,  V1, sz);
                scaling_v_border<<<1024,1024,1024>>>(l_ranks, l_border, V,  V2, sz1 );
                cudaDeviceSynchronize();
                cudaEventRecord(tstop,0);
                cudaEventSynchronize(tstop);
                cudaEventElapsedTime(&elapsed_gpu, tstart, tstop);
                cudaEventDestroy(tstart);
                cudaEventDestroy(tstop);
                printf("\nGPU time for doing Scaling(OLD nodes and Border nodes): %f (msec)\n", elapsed_gpu);
                cout<<"Total No. of threads used for Scaling: "<<sz+sz1<<endl;
                cudaFree(l_border);
                cudaFree(l_old);
                time_spent+=elapsed_gpu;
                time_spent1+=elapsed_gpu;
                double error=0.0;
                double margin_of_error=0.00000001;
                double trans_elapsed1=0.00;
                double *dest0,*l_ranks2;
                int *l_temp,*out_link;
                int *l_pred2;
                double *dest5;
                double *dest4;
                dest4 = (double*)calloc(  z, sizeof(double) ); 
                cout<<"ji"<<endl;
                double d=0.85;
                int nvpg=z;
                int l_size=z;
                int l_sz=root.size();
                double *old_ranks;
                old_ranks=(double*)malloc( ranks.size() * sizeof(double) );
                double *new_ranks;
                cudaDeviceSynchronize();
                for(int t=0;t<1;t++)
                {       
                        if(error<margin_of_error)
                        {
                                double euc=0.0;
				
                                std::copy(ranks.begin(), ranks.end(), old_ranks);
                                //cout<<"1"<<endl;
                                int i=0;
                                cudaMalloc(&l_temp, C*R*sizeof(int));
                                cudaMalloc(&out_link, C*R*sizeof(int));
                                cudaMalloc(&l_pred2, 1000*ranks.size()*sizeof(int));
                                cudaMalloc(&dest0, z*sizeof(double));
                                cudaMalloc(&dest5, z*sizeof(double));
                                //cout<<"2"<<endl;
                                auto trans3 = chrono::steady_clock::now();
                                if(cudaMemcpy(l_temp, vtemp1, z*sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
                                {
                                        printf("Error 3 in cudaMemcpy\n");
                                        return -2;
                                }
                                
                                if(cudaMemcpy(out_link, outll, C*R*sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
                                {
                                        printf("Error 4 in cudaMemcpy\n");
                                        return -2;
                                }
                                
                                if(cudaMemcpy(l_pred2, pred2, (ranks.size())*sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
                                {
                                        printf("Error 5 in cudaMemcpy\n");
                                        return -2;
                                }
                                auto trans4 = chrono::steady_clock::now();
                                trans_elapsed1 = chrono::duration_cast<chrono::milliseconds>(trans4 - trans3).count();
                                //cout<<trans_elapsed1<<endl;
                                time_spent2+=trans_elapsed1;
                                //cout<<"3"<<endl;
                                cudaEventCreate(&tstart);
                                cudaEventCreate(&tstop);
                                cudaEventRecord(tstart, 0);
                                ranks_sum_cal<<<1024,1024,1024>>>(dest0,l_temp,out_link,l_pred2,l_ranks,l_size,l_sz);
                                pagerank_aff<<<1024,1024,1024>>>(dest5, dest0, d, nvpg ,l_size);
                                //cout<<"4"<<endl;
                                cudaDeviceSynchronize();
                                cudaEventRecord(tstop,0);
                                cudaEventSynchronize(tstop);
                                cudaEventElapsedTime(&elapsed_gpu, tstart, tstop);
                                cudaEventDestroy(tstart);
                                cudaEventDestroy(tstop);
                                time_spent+=elapsed_gpu;
                                time_spent1+=elapsed_gpu;
                                cudaMemcpy( dest4, dest5, z*sizeof(double),cudaMemcpyDeviceToHost );
                                //cout<<"5"<<endl;
                                for(int i =0;i<z;i++)
                                {
                                        ranks.at(vtemp1[i])=dest4[i];
                                }
                                //cout<<"6"<<endl;
                                int NUM=ranks.size();
                                float j=0.0;
                                for (int i = 0 ; i < NUM ; ++i)
                                {
                                        j+=ranks[i];
                                }
                                //cout<<"7"<<endl;
                                new_ranks=(double*)calloc( ranks.size() ,sizeof(double) );
                                for (int i = 0 ; i < NUM ; ++i)
                                {
                                        ranks[i]=ranks[i]/j;
                                        new_ranks[i] = ranks[i];
                                        euc = pow((new_ranks[i] - old_ranks[i]),2);
                                        error+=euc;   
                                }
                                //cout<<"8"<<endl;
                                free(new_ranks);
                                free(old_ranks);
                                cudaFree(l_temp);
                                cudaFree(out_link);
                                cudaFree(l_pred2);
                        }
                        else
                        {
                                break;
                        }
                }
                printf("\nGPU time(NEW nodes): %f (msec)\n", elapsed_gpu);
                cout<<"Total No. of threads used for PR: "<<l_sz+l_size<<endl;
                cout<<"TOTAL GPU Time elapsed for current batch: "<<time_spent1<<" msec"<<endl;
                cout<<"TOTAL Transfer time: "<<time_spent2<<" msec"<<endl;
                float j1=0.0;
                cout << "Sum of Ranks = "<< accumulate(ranks.begin(), ranks.end(), j1)<<endl;
                cout<<"********************************************************************************************************************************"<<endl;
                cudaFree(dest0);
                cudaFree(dest5);
                cudaFree(l_ranks);
                free(vtemp1);
                free(vborder);
                free(times);
                free(succ1);
                free(pred1);
                free(outll);
                free(pred2);
                free(ranks1);
                free(vold);
                free(vnew);
                free(dest4);
                /*cudaDeviceReset();
                cudaGetErrorString(cudaGetLastError());    
                printf("Sync: %s\n", cudaGetErrorString(cudaDeviceSynchronize()));*/
        }
        cout<<"\nTotal time elapsed on GPU for all batches: "<<time_spent<<" msec";
        return 0;
}
