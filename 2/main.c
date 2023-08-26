#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <stdbool.h>
#include <sys/shm.h>

void makeOffspring(){
    for(int i=0;i<20;i++)
    {
        int j=fork();
        if(j==0)
        {
            printf("Jsem potomek %d\n",i);
            sleep(i+1);
            //printf("Jsem potomek %d\n",i);
            exit(i+2);
        }
    }
    int status;
    int returncodes[20];
    int i = 0;
    while(waitpid(-1, &status, 0) > 0 ){
        //printf("Potomek s navratovou hodnotou %d ukoncen\n", WEXITSTATUS(status));
        
    printf("Navratove hodnoty potomku jsou: ");
    for(i = 0; i<20; i++){
        printf("%d ", returncodes[i]);
    }
}

void swap(int* xp, int* yp)
{
    int temp = *xp;
    *xp = *yp;
    *yp = temp;
}
 
void bubbleSort(int arr[], int size)
{
    int status;
    bool isSorted = false;
    while(isSorted == false){
        isSorted = true; // pokud se nic nepřehodí, pole je setříděné -> vždy proběhne minimálně jedna ověřovací iterace
        for (int i=1; i<=size-2; i=i+2)
        {
            int j=fork();
            if(j==0)
            {
                //printf("Jsem potomek %d\n",i);
                if (arr[i] > arr[i+1])
                {
                    swap(&arr[i], &arr[i+1]);
                    exit(1);
                }
                exit(0);
            }
        }
        while(waitpid(-1, &status, 0) > 0 ){ // kontroluju návratové hodnoty, dalo by se řešit taky pomocí shared memory booleanu
            if(WEXITSTATUS(status) == 1){
                isSorted = false;
            }
        }
        for (int i=0; i<=size-2; i=i+2)
        {
            int j=fork();
            if(j==0)
            {
                printf("Jsem potomek %d\n",i);
                if (arr[i] > arr[i+1])
                {
                    swap(&arr[i], &arr[i+1]);
                    exit(1);
                }
                exit(0);
            }
        }
        while(waitpid(-1, &status, 0) > 0 ){
            if(WEXITSTATUS(status) == 1){
                isSorted = false;
            }
        }
    }
}
 
void printArray(int arr[], int size)
{
    int i;
    for (i = 0; i < size; i++)
        printf("%d ", arr[i]);
    printf("\n");
}

int main(){
    //makeOffspring();
    int shmid;   //identifikátor paměti
    int *shmp;  //ukazatel na sdílenou paměť
    int shmkey = getuid();  //identifikátor sdílené paměti, program getuid vrátí pro každého uživatele jiné unikátní číslo proto je pro identifikátor sdílené paměti vhodný, samozřejmě by šlo napsat i libovolné statické číslo, ale pokud by někdo použil v programu stejné číslo "popralo by se to"

    shmid = shmget(shmkey, 10*sizeof(int), 0644|IPC_CREAT);  //vytvoření sdílené paměti o velikosti  10 intů, tedy ekvivalent int shmp[10]; 
    shmp  = shmat(shmid,NULL,0);  // tady se sváže ukazatel se sdílenou pamětí
    shmp[0] = 141;
    shmp[1] = 154;
    shmp[2] = 1;
    shmp[3] = 2;
    shmp[4] = 89;
    shmp[5] = 35;
    shmp[6] = 27;
    shmp[7] = 185;
    shmp[8] = 98;
    shmp[9] = 25;
    //{141,154,1,2,89,35,27,185,98,25};
    bubbleSort(shmp,10);
    printf("Setridene pole: \n");
    printArray(shmp,10);
    return 0;
}