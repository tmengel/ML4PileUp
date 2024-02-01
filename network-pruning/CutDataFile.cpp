#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TMath.h"

#include <iostream>
#include <vector>

using namespace std;
using namespace ROOT;

int main(int argc, char** argv)
{
    TString inputfile = "/lustre/isaac/scratch/tmengel/ML4PileUp/data/DataSmall.root";
    std::cout << "inputfile: " << inputfile.Data() << std::endl;

    TString treename = "OutputTree";
    std::cout << "treename: " << treename.Data() << std::endl;

    // open file
    TFile fin(inputfile.Data(), "READ");
    if(!fin.IsOpen()){
        std::cout << "Error: could not open file " << inputfile << std::endl;
        return 1;
    }
    TTree * tree = (TTree*)fin.Get(treename.Data());
    if(!tree){
        std::cout << "Error: could not find tree in file " << inputfile << std::endl;
        return 1;
    }

    // input variables
    double amp = 0;
    double phase = 0;
    std::vector<double> * trace = 0;

    // set branch addresses
    tree->SetBranchAddress("amp",&amp);
    tree->SetBranchAddress("phase",&phase);
    tree->SetBranchAddress("trace",&trace);

    int nentries = tree->GetEntries();

    // output variables
    int pile_out = 0;
    float amp_out = 0;
    float phase_out = 0;
    std::vector<float> trace_out;


    // output file
    TString outputfile = "DataSmallFloat.root";
    TFile * fout = new TFile(outputfile.Data(), "RECREATE");
    TTree * tree_out = new TTree("OutputTree", "OutputTree");
    // tree_out->Branch("pile", &pile_out);
    tree_out->Branch("amp", &amp_out);
    tree_out->Branch("phase", &phase_out);
    tree_out->Branch("trace", &trace_out);

    // loop over events
    // int nEventsToSave = 10000;
    int events_saved = 0;
    for(int i = 0; i < nentries; i++){
        
        if(i % 10 != 0) continue;
        tree->GetEntry(i);

        if(i % 1000000 == 0) std::cout << "Event " << i << " of " << nentries << std::endl;

        // only save events with no pileup
        // if (phase == 0.0) continue;
        // events_saved++;
        amp_out = amp;
        phase_out = phase;
        // pile_out = 0;
        trace_out.clear();
        for(int j = 0; j < trace->size(); j++){
            trace_out.push_back(trace->at(j));
        }

        tree_out->Fill();

        // if(i > nEventsToSave) break;
    }

    fout->Write();
    fout->Close();

    // clean up
    fin.Close();

    std::cout << "Done!" << std::endl;


    return 0;

}