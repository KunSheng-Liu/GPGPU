/**
 * \name    main.cpp
 * 
 * \brief   This code instance a GPGPU for simulating the GPGPU behavior
 * 
 * \date    APR 18, 2023
 */

#include "include/App_config.h"
#include "include/Log.h"

#include "include/GPGPU.hpp"

/* ************************************************************************************************
 * Global Variable
 * ************************************************************************************************
 */
Command command;
string program_name;

/* ************************************************************************************************
 * Main
 * ************************************************************************************************
 */
void parser_cmd (int argc, char** argv);

int main (int argc, char** argv)
{
    parser_cmd(argc, argv);

    timeval start, end;
    gettimeofday(&start, NULL);

        std::cout << "Hello GPGPU" << std::endl;

        GPGPU mGPGPU;

        mGPGPU.run();

        std::cout << "GPGPU Done!" << std::endl;

    gettimeofday(&end, NULL);
    std::cout << "Total spend time: " << to_string((1000000 * (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)) * 0.001) << " ms" << std::endl;
    

    return 0;
}

void parser_cmd (int argc, char** argv) 
{
    string scheduler_name = "Greedy";
    string batch_name     = "Disable";
    string mem_name       = "None";
    
    for (int i = 1; i < argc;)
    {
        string flag = argv[i++];
        if (flag == "-S" || flag == "--scheduler") 
        {
            try{
                string option = argv[i++];
                if (option == "Greedy")        command.SCHEDULER_MODE = SCHEDULER::Greedy;
                else if (option == "Baseline") command.SCHEDULER_MODE = SCHEDULER::Baseline;
                else if (option == "BARM")     command.SCHEDULER_MODE = SCHEDULER::BARM;
                else if (option == "LazyB")    command.SCHEDULER_MODE = SCHEDULER::LazyB;
                else if (option == "My")       command.SCHEDULER_MODE = SCHEDULER::My;
                else ASSERT(false, "Wrong argument, try --help");
                scheduler_name = option;
            } 
            catch(exception e) ASSERT(false, "Wrong argument, try --help");
        }
        // else if (flag == "-I" || flag == "--inference-method")
        // {
        //     try{
        //         string option = argv[i++];
        //         if (option == "Sequential")    command.INFERENCE_MODE = INFERENCE_TYPE::SEQUENTIAL;
        //         else if (option == "Parallel") command.INFERENCE_MODE = INFERENCE_TYPE::PARALLEL;
        //         else ASSERT(false, "Wrong argument, try --help");
        //         inference_name = option;
        //     } 
        //     catch(exception e) ASSERT(false, "Wrong argument, try --help");
            
        // }
        else if (flag == "-B" || flag == "--batch-inference") 
        {
            try{
                string option = argv[i++];
                if (option == "Disable")       command.BATCH_MODE = BATCH_METHOD::DISABLE;
                else if (option == "Max")      command.BATCH_MODE = BATCH_METHOD::MAX;
                else ASSERT(false, "Wrong argument, try --help");
                batch_name = option;
            } 
            catch(exception e) ASSERT(false, "Wrong argument, try --help");
        }
        else if (flag == "-M" || flag == "--mem-allocate") 
        {
            try{
                string option = argv[i++];
                if (option == "None")          command.MEM_MODE = MEM_ALLOCATION::None;
                else if (option == "Average")  command.MEM_MODE = MEM_ALLOCATION::Average;
                else if (option == "MEMA")     command.MEM_MODE = MEM_ALLOCATION::MEMA;
                else if (option == "R_MEMA")   command.MEM_MODE = MEM_ALLOCATION::R_MEMA;
                else ASSERT(false, "Wrong argument, try --help");
                mem_name = option;
            } 
            catch(exception e) ASSERT(false, "Wrong argument, try --help");
        }
        else if (flag == "-T" || flag == "--test-set") 
        {
            try{
                string option = argv[i++];
                float batch_size   = atof(argv[i++]);
                float arrival_time = atof(argv[i++]);
                float period       = atof(argv[i++]);
                float deadline     = atof(argv[i++]);
                auto task_config = make_tuple(batch_size, arrival_time, period, deadline);
                
                if (option == "LeNet")          command.TASK_LIST.emplace_back(make_pair(APPLICATION::LeNet,     task_config));
                else if (option == "CaffeNet")  command.TASK_LIST.emplace_back(make_pair(APPLICATION::CaffeNet,  task_config));
                else if (option == "ResNet18")  command.TASK_LIST.emplace_back(make_pair(APPLICATION::ResNet18,  task_config));
                else if (option == "GoogleNet") command.TASK_LIST.emplace_back(make_pair(APPLICATION::GoogleNet, task_config));
                else if (option == "VGG16")     command.TASK_LIST.emplace_back(make_pair(APPLICATION::VGG16,     task_config));
                else if (option == "All")       command.TASK_LIST.emplace_back(make_pair(APPLICATION::ALL,       task_config));
                else if (option == "Light")     command.TASK_LIST.emplace_back(make_pair(APPLICATION::LIGHT,     task_config));
                else if (option == "Heavy")     command.TASK_LIST.emplace_back(make_pair(APPLICATION::HEAVY,     task_config));
                else if (option == "Test1")     command.TASK_LIST.emplace_back(make_pair(APPLICATION::TEST1,     task_config));
                else if (option == "Test2")     command.TASK_LIST.emplace_back(make_pair(APPLICATION::TEST2,     task_config));
                else ASSERT(false, "Wrong argument, try --help");
            } 
            catch(exception e) ASSERT(false, "Wrong argument, try --help");
            
        }
        else if (flag == "-h" || flag == "--help") {
            std::cout << "GPGPU: GPGPU [[-S | -I | -M | -T] [OPTION]]" << std::endl;

            std::cout << "Detial:" << std::endl;
            std::cout << "\t-S, " << std::left << setw(20) << "--scheduler"        << "Greedy | Baseline | BARM | LazyB" << std::endl;
            std::cout << "\t-I, " << std::left << setw(20) << "--inference-method" << "Sequential | Parallel"            << std::endl;
            std::cout << "\t-B, " << std::left << setw(20) << "--batch-inference"  << "Disable | Max"                    << std::endl;
            std::cout << "\t-M, " << std::left << setw(20) << "--mem-allocate"     << "None | Average | MEMA | R_MEMA"   << std::endl;
            std::cout << "\t-T, " << std::left << setw(20) << "--test-set"         << "LeNet | CaffeNet | ResNet18 | GoogleNet | VGG16 | Light | Heavy | Mix | All | Test1 | Test2"  << std::endl;

            std::cout << "Examples:" << std::endl;
            std::cout << "\t./GPGPU" << std::endl;
            std::cout << "\t./GPGPU -I Sequential -T ResNet18 3 0 -1 100 -T VGG16 1 0 10 100 -T GoogleNet 2 0 2 10 " << std::endl;
            std::cout << "\t./GPGPU -sm-dispatch Baseline -M Average" << std::endl;

            std::cout << "Default:" << std::endl;
            std::cout << "./GPGPU -S Greedy -I Sequential -B Disable -M None -T NULL" << std::endl;

            exit(1);
        } else ASSERT(false, "Wrong argument, try --help");
    }

    program_name = scheduler_name + "_" + batch_name + "_" + mem_name;
}