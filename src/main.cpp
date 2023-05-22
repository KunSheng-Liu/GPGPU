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
    string inference_name = "Sequential";
    string batch_name     = "Disable";
    string sm_name        = "Greedy";
    string mem_name       = "None";
    
    for (int i = 1; i < argc;)
    {
        string flag = argv[i++];
        if (flag == "-I" || flag == "--inference-method")
        {
            try{
                string option = argv[i++];
                if (option == "Sequential")    command.INFERENCE_MODE = INFERENCE_TYPE::SEQUENTIAL;
                else if (option == "Parallel") command.INFERENCE_MODE = INFERENCE_TYPE::PARALLEL;
                else ASSERT(false, "Wrong argument, try --help");
                inference_name = option;
            } 
            catch(exception e) ASSERT(false, "Wrong argument, try --help");
            
        }
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
        else if (flag == "-S" || flag == "--sm-dispatch") 
        {
            try{
                string option = argv[i++];
                if (option == "Greedy")        command.SM_MODE = SM_DISPATCH::Greedy;
                else if (option == "Baseline") command.SM_MODE = SM_DISPATCH::Baseline;
                else if (option == "Equal")    command.SM_MODE = SM_DISPATCH::Equal;
                else if (option == "SMD")      command.SM_MODE = SM_DISPATCH::SMD;
                else if (option == "R_SMD")    command.SM_MODE = SM_DISPATCH::R_SMD;
                else ASSERT(false, "Wrong argument, try --help");
                sm_name = option;
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
                if (option == "Light")          command.TASK_MODE = TASK_SET::LIGHT;
                else if (option == "Heavy")     command.TASK_MODE = TASK_SET::HEAVY;
                else if (option == "All")       command.TASK_MODE = TASK_SET::ALL;
                else if (option == "LeNet")     command.TASK_MODE = TASK_SET::LeNet;
                else if (option == "CaffeNet")  command.TASK_MODE = TASK_SET::CaffeNet;
                else if (option == "ResNet18")  command.TASK_MODE = TASK_SET::ResNet18;
                else if (option == "VGG16")     command.TASK_MODE = TASK_SET::VGG16;
                else if (option == "GoogleNet") command.TASK_MODE = TASK_SET::GoogleNet;
                else if (option == "Test1")     command.TASK_MODE = TASK_SET::TEST1;
                else if (option == "Test2")     command.TASK_MODE = TASK_SET::TEST2;
                else ASSERT(false, "Wrong argument, try --help");
            } 
            catch(exception e) ASSERT(false, "Wrong argument, try --help");
            
        }
        else if (flag == "-h" || flag == "--help") {
            std::cout << "GPGPU: GPGPU [[-I | -S | -M | -T] [OPTION]]" << std::endl;

            std::cout << "Detial:" << std::endl;
            std::cout << "\t-I, " << std::left << setw(20) << "--inference-method" << "Sequential | Parallel"            << std::endl;
            std::cout << "\t-B, " << std::left << setw(20) << "--batch-inference"  << "Disable | Max"                    << std::endl;
            std::cout << "\t-S, " << std::left << setw(20) << "--sm-dispatch"      << "Greedy | Baseline | Equal | SMD | R_SMD"   << std::endl;
            std::cout << "\t-M, " << std::left << setw(20) << "--mem-allocate"     << "None | Average | MEMA | R_MEMA"   << std::endl;
            std::cout << "\t-T, " << std::left << setw(20) << "--test-set"         << "Light | Heavy | Mix | All | LeNet | CaffeNet | ResNet18 | VGG16 | GoogleNet | Test1 | Test2"  << std::endl;

            std::cout << "Examples:" << std::endl;
            std::cout << "\t./GPGPU" << std::endl;
            std::cout << "\t./GPGPU -I Sequential -T Heavy" << std::endl;
            std::cout << "\t./GPGPU -sm-dispatch Average -M Average" << std::endl;

            std::cout << "Default:" << std::endl;
            std::cout << "./GPGPU -I Sequential -B Disable -S Baseline -M None -T Light" << endl;

            exit(1);
        } else ASSERT(false, "Wrong argument, try --help");
    }

    program_name = inference_name + "_" + batch_name + "_" + sm_name + "_" + mem_name;
}