#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>

#include <TFile.h>
#include <TTree.h>
#include <TString.h>
#include <TMath.h>


int main(int argc, char** argv)
{   
    if(argc != 2)
    {
        std::cout << "Usage: CutDataFile <inputfile>" << std::endl;
        return 1;
    }
    
    std::string inputfile = argv[1];
    std::string outputfile = "DataSmallFloat.root";
    if(argc == 3) outputfile = argv[2];
    float percentage = 0.1;
    if(argc == 4) percentage = atof(argv[3]);

    bool pile_up_only = false;
    if(argc == 5) pile_up_only = atoi(argv[4]) == 1 ? true : false;

    // print input
    std::cout << "inputfile: " << inputfile << std::endl;
    std::cout << "outputfile: " << outputfile << std::endl;
    std::cout << "percentage: " << percentage << std::endl;
    std::cout << "pile_up_only: " << pile_up_only << std::endl;

    // tree name
    std::string treename = "OutputTree";

    // open file
    TFile * fin = new TFile(inputfile.c_str(), "READ");
    if(!fin->IsOpen()){
        std::cout << "Error: could not open file " << inputfile << std::endl;
        exit(1); // failed
    }

    TTree * tree = (TTree*)fin->Get(treename.c_str());
    if(!tree){
        std::cout << "Error: could not find tree in file " << inputfile << std::endl;
        exit(1); // failed
    }

    // input variables
    bool pileup = false;
    double amp = 0;
    double phase = 0;
    std::vector<double> * trace = 0;

    // set branch addresses
    tree->SetBranchAddress("pileup",&pileup);
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
    TFile * fout = new TFile(outputfile.c_str(), "RECREATE");
    TTree * tree_out = new TTree("OutputTree", "OutputTree");
    tree_out->Branch("pileup", &pile_out);
    tree_out->Branch("amp", &amp_out);
    tree_out->Branch("phase", &phase_out);
    tree_out->Branch("trace", &trace_out);

    int events_saved = 0;
    int inverted_percentage = int(1.0/percentage);

    for(int i = 0; i < nentries; i++)
    {
        
        if(i % inverted_percentage != 0) continue; // skip events
        if(i % 1000000 == 0) std::cout << "Event " << i << " of " << nentries << std::endl;

        tree->GetEntry(i);

        // only save events with no pileup
        if(pile_up_only && !pileup) continue;
    
        amp_out = amp;
        phase_out = phase;
        pile_out = pileup ? 1 : 0;
        trace_out.clear();

        for(int j = 0; j < trace->size(); j++)
        {
            trace_out.push_back(trace->at(j));
        }

        tree_out->Fill();
        events_saved++;
    }

    std::cout << "Events saved: " << events_saved << std::endl;

    fout->Write();
    fout->Close();

    // clean up
    fin->Close();

    std::cout << "Done!" << std::endl;


    return 0;
   
}