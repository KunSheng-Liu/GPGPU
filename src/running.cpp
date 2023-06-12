/**
 * \name    Running.cpp
 * 
 * \brief   It's a thread running tool
 * 
 * \date    May 18, 2023
 */

#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <semaphore.h>

/* ************************************************************************************************
 * Name Space
 * ************************************************************************************************
 */
using namespace std;

/* ************************************************************************************************
 * Software Configuration
 * ************************************************************************************************
 */
#define THREAD_NUM_LIMIT 4

/* ************************************************************************************************
 * Function Declaration
 * ************************************************************************************************
 */
void exe_thread(string cmd);

/* ************************************************************************************************
 * Global Resource
 * ************************************************************************************************
 */
sem_t* count_semaphore = new sem_t;

/** ===============================================================================================
 * \name    main
 * 
 * \brief   Thread executing the command inside the \b ./taskset_list.txt file
 * 
 * \param   numOfThread    the maximum parallel thread can execute
 * 
 * \note    can ignore the command by add "//" in the start of line
 * 
 * \endcond
 * ================================================================================================
 */
int main(int argc, char** argv)
{
    vector<string> task_list;
    
    /* Load the file */
    fstream file;   
    file.open("./taskset_list.txt", ios::in); 
    if (!file.is_open()) {cout << "ERROR Can't open file\n"; abort();}

    /* Parser vaild command */
    string readLine;
    while (getline(file, readLine))
    {
        if (!(readLine.find("//") == 0 || readLine.empty())) task_list.push_back(readLine);
    }

    /* Determine the parallel ability */
    int thread_num = (argc > 1) ? atoi(argv[1]) : THREAD_NUM_LIMIT;
    sem_init(count_semaphore, 0, thread_num);

    /* Launch tasks */
    thread threads[task_list.size()];
    for (int i = 0; i < task_list.size(); i++)
    {
        threads[i] = thread(exe_thread, task_list.at(i));
    }

    /* Waiting thread finish */
    for (int i = 0; i < task_list.size(); i++)
    {
        threads[i].join();
    }
    
    sem_destroy(count_semaphore);
}


/** ===============================================================================================
 * \name    exe_thread
 * 
 * \brief   Thread executing the command once the thread pend the semaphore
 * 
 * \param   cmd    the command going to execute
 * 
 * \endcond
 * ================================================================================================
 */
void exe_thread(string cmd)
{
    sem_wait(count_semaphore);
        std::cout << "Exec cmd: " << cmd << std::endl;
        int ret = system(cmd.c_str());
    sem_post(count_semaphore);

    std::cout << "Finish: " << cmd << std::endl;
}
