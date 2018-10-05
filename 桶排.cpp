#include <iostream>
using namespace std;
int n,tp[10000];                          //桶排数组,需初始化 
int main()
{
    int i,j;
    cin>>n;                 //读入n
    for(i=1;i<=n;i++)       //读入n个数据
    { 
        cin>>j;   
        tp[j]++;            //用数组下标进行排序 
    }                                                  //桶排：将n个输入元素分配到这些桶中，对桶中元素进行排序                                            
    for(i=1;i<=n;i++)                                  //时间复杂度为O(N)，空间复杂度为O(N)  
    {                                                  //特点：快，但空间浪费严重，要求数据连续并小等n
        while(tp[i])
        {
            cout<<i<<" ";
            tp[i]--;        //处理重复数据
        }
    }
    return 0;               //若有负数，数组开两倍n处理即可
}                                                                                                 
                                                      
    
