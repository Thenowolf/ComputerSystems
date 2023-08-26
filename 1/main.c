#include <stdio.h>
 
void swap(int* xp, int* yp)
{
    int temp = *xp;
    *xp = *yp;
    *yp = temp;
}
 

void bubbleSort(int arr[], int size)
{
    for (int i = 0; i < size - 1; i++) // porovnávám MEZI čísly, tedy -1
    {
        for (int j = 0; j < size - i - 1; j++) //neporovnávám už, to, co jsem porovnal, proto -i
        {
            if (arr[j] > arr[j + 1])
            {
                swap(&arr[j], &arr[j + 1]);
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
 
int main()
{
    int arr[] = { 5, 4, 3, 2, 1 };
    int size = sizeof(arr) / sizeof(arr[0]);
    printf("Puvodni pole: \n");
    printArray(arr,size);
    bubbleSort(arr,size);
    printf("Setridene pole: \n");
    printArray(arr,size);
    return 0;
}