#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

#include "hl_Core.h"
#include "hl_FieldStruct.h"
#include "hl_IntFieldStruct.h"
#include "hl_Timer.h"

#include "Block_Map.h"
#include "GenericFieldStruct.h"

using namespace hiperlife;

int function(int step, int iField, int index)
{
    return step * 3 + index + iField * 7;
}

template<typename T> bool checkGhosts(T* field, int step, std::vector<int>& listGhosts)
{
    for (int ghost : listGhosts)
    {
        for (int iField = 0; iField < field->numFlds(); iField++)
        {
            int expectedValue = function(step, iField, ghost);
            int actualValue = field->getValue(iField, ghost, IndexType::Global);
            if (expectedValue != actualValue)
            {
                std::cerr << MyRank() << ": mismatch at ghost index " << ghost << ", field " << iField
                          << ": expected " << expectedValue << ", got " << actualValue << std::endl;
                return false;
            }
        }
    }
    return true;

}

std::pair<std::vector<int>, std::vector<int>> GenerateInitialData(int numItemsPerCore, int numFields)
{
    using std::vector;
    const int myRank = hiperlife::MyRank();

    std::mt19937 gen(1234+myRank);

    std::uniform_int_distribution<int> sizeDist(static_cast<int>(0.75 * numItemsPerCore), static_cast<int>(1.25 * numItemsPerCore));
    int localItemSize = sizeDist(gen);
    vector<int> distributedData(localItemSize * numFields);

    int nItems{};
    MPI_Allreduce(&localItemSize, &nItems, 1, MPI_INT, MPI_SUM, hiperlife::MpiComm());
    int offset{};
    MPI_Exscan(&localItemSize, &offset, 1, MPI_INT, MPI_SUM, hiperlife::MpiComm());

    std::uniform_real_distribution<double> valueDist(0.0, 1.0);
    for (int i = 0; i < localItemSize; ++i)
        for (int d = 0; d < numFields; ++d)
            distributedData[d*localItemSize + i] = function(0, d, offset+i);


    int localGhostSize = static_cast<int>(0.25 * localItemSize);
    std::vector<int> listGhosts(localGhostSize);
    for (int i = 0; i < localGhostSize; ++i) {
        listGhosts[i] = static_cast<int>(valueDist(gen) * nItems);
    }

    return {distributedData, listGhosts};
}

SmartPtr<GenericFieldStruct<int>> CreateFieldNew(int numFields, std::vector<int>& distributedData)
{
    using namespace hiperlife;

    const int localItemSize = static_cast<int>(distributedData.size() / numFields);
    BlockMap myBlockMap("MyMap", MpiComm());
    myBlockMap.setLocNItem(localItemSize);
    myBlockMap.Update();

    auto fieldExplicit = Create<GenericFieldStruct<int>>("FieldExplicit", MpiComm());
    fieldExplicit->setBlockMap(myBlockMap);
    fieldExplicit->setDistFields(distributedData);
    fieldExplicit->setNFlds(numFields);
    fieldExplicit->Update();

    return fieldExplicit;
}

SmartPtr<hiperlife::IntFieldStruct> CreateFieldOld(int locItems, int items, int numFields, std::vector<int>& distributedData)
{
    using namespace hiperlife;

    auto epetraComm = Create<Epetra_MpiComm>(hiperlife::MpiComm());
    auto epetraMap = Create<Epetra_BlockMap>(items, locItems, 1, 0, *epetraComm);
    int* ptr = distributedData.data();

    auto oldField = Create<IntFieldStruct>("OldField", MpiComm());
    oldField->setBlockMap(*epetraMap);
    oldField->setDistFields(ptr);
    oldField->setNFlds(numFields);
    oldField->Update();

    return oldField;
}

int main(int argc, char** argv)
{
    using std::vector;
    using namespace hiperlife;

    const int numFields = 2;
    const int numItemsPerCore = 100;

    Init(argc, argv);

    auto [distributedData, listGhosts] = GenerateInitialData(numItemsPerCore, numFields);
    int localItemSize = static_cast<int>(distributedData.size() / numFields);
    int nItems;
    MPI_Allreduce(&localItemSize, &nItems, 1, MPI_INT, MPI_SUM, hiperlife::MpiComm());

    timer::start("NewUpdate");
    auto fieldExplicit = CreateFieldNew(numFields, distributedData);
    timer::stop("NewUpdate");

    timer::start("NewCreateGhostSt");
    fieldExplicit->UpdateGhosts(listGhosts);
    timer::stop("NewCreateGhostSt");

    timer::start("NewCommGhost");
    fieldExplicit->UpdateGhosts();
    timer::stop("NewCommGhost");
    if (VerboseRank()) std::cout << "New field" << std::endl;
    GlobalBarrier();
    bool ok = checkGhosts(fieldExplicit.get(), 0, listGhosts);
    //if (!ok) {
    //    hiperlife::Abort("Ghost values in new field do not match expected values.");
    //}
    GlobalBarrier();

    timer::start("OldUpdate");
    auto oldField = CreateFieldOld(localItemSize, nItems, numFields, distributedData);
    timer::stop("OldUpdate");

    timer::start("OldCreateGhostSt");
    oldField->UpdateGhosts(listGhosts);
    timer::stop("OldCreateGhostSt");

    timer::start("OldCommGhost");
    oldField->UpdateGhosts();
    timer::stop("OldCommGhost");
    if (VerboseRank()) std::cout << "oldField" << std::endl;
    GlobalBarrier();
    ok = checkGhosts(oldField.get(), 0, listGhosts);
    GlobalBarrier();
    //if (!ok) {
    //    hiperlife::Abort("Ghost values in old field do not match expected values.");
    //}

    hiperlife::GlobalBarrier();

    std::cout << *fieldExplicit << std::endl;
    std::cout << *oldField << std::endl;

    hiperlife::GlobalBarrier();

    timer::output();

    if (VerboseRank()) {
        std::cout << "Improvement in Update:        " << timer::getTime("OldUpdate") / timer::getTime("NewUpdate") << "x" << std::endl;
        std::cout << "Improvement in CreateGhostSt: " << timer::getTime("OldCreateGhostSt") / timer::getTime("NewCreateGhostSt") << "x" << std::endl;
        std::cout << "Improvement in CommGhost:     " << timer::getTime("OldCommGhost") / timer::getTime("NewCommGhost") << "x" << std::endl;
        std::cout << "Last OldCommGhost:     " << timer::getTime("OldCommGhost") << std::endl;
    }

    Finalize();
    return 0;
}

