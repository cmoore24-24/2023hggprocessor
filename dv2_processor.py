from coffea.nanoevents import NanoEventsFactory, BaseSchema, PFNanoAODSchema
import json
import fastjet
import numpy as np
import awkward as ak
from coffea import processor
import hist
import coffea.nanoevents.methods.vector as vector
import warnings
import hist.dask as dhist
import dask
import pickle
import os
import distributed
from ndcctools.taskvine import DaskVine
import time

full_start = time.time()

if __name__ == '__main__':
    m = DaskVine([9123,9128], name="hgg", run_info_path='/project01/ndcms/cmoore24/vine-run-info')

    warnings.filterwarnings("ignore", "Found duplicate branch")
    warnings.filterwarnings("ignore", "Missing cross-reference index for")
    warnings.filterwarnings("ignore", "dcut")
    warnings.filterwarnings("ignore", "Please ensure")
    warnings.filterwarnings("ignore", "The necessary")

    try:
        with open("data_dv2.pkl", "rb") as fr:
            datasets = pickle.load(fr)
    except Exception as e:
        print(f"Could not read data_dv2.pkl: {e}, reconstructing...")
        datasets = {
            "hgg": {"path": "signal/hgg", "label": "Hgg"},
            "hbb": {"path": "signal/hbb", "label": "Hbb"},
            "q347": {"path": "qcd/300to470", "label": "QCD_Pt_300to470"},
            "q476": {"path": "qcd/470to600", "label": "QCD_Pt_470to600"},
            "q68": {"path": "qcd/600to800", "label": "QCD_Pt_600to800"},
            "q810": {"path": "qcd/800to1000", "label": "QCD_Pt_800to1000"},
            "q1014": {"path": "qcd/1000to1400", "label": "QCD_Pt_1000to1400"},
            "q1418": {"path": "qcd/1400to1800", "label": "QCD_Pt_1400to1800"},
            "q1824": {"path": "qcd/1800to2400", "label": "QCD_Pt_1800to2400"},
            "q2432": {"path": "qcd/2400to3200", "label": "QCD_Pt_2400to3200"},
            "q32Inf": {"path": "qcd/3200toInf", "label": "QCD_Pt_3200toInf"},
        }

        path_root = "/project01/ndcms/cmoore24"
        for name, info in datasets.items():
            info['files'] = os.listdir(f"{path_root}/{info['path']}")

        with open("data_dv2.pkl", "wb") as fw:
            pickle.dump(datasets, fw)

    source = "/project01/ndcms/cmoore24"
    events = {}
    for name, info in datasets.items():
        events[name] = hgg = NanoEventsFactory.from_root(
            {f"{source}/{info['path']}/{fn}": "/Events" for fn in info["files"]},
            permit_dask=True,
            schemaclass=PFNanoAODSchema,
            chunks_per_file=1,
            metadata={"dataset": info["label"]},
        ).events()

    def color_ring(fatjet):
        jetdef = fastjet.JetDefinition(fastjet.cambridge_algorithm, 0.8) # make this C/A at 0.8
        pf = ak.flatten(fatjet.constituents.pf, axis=1)
        cluster = fastjet.ClusterSequence(pf, jetdef)
        subjets = cluster.exclusive_subjets_up_to(data=cluster.exclusive_jets(n_jets=1), nsub=3) #uncomment this when using C/A
        #subjets = cluster.inclusive_jets()
        vec = ak.zip({
            "x": subjets.px,
            "y": subjets.py,
            "z": subjets.pz,
            "t": subjets.E,
            },
            with_name = "LorentzVector",
            behavior=vector.behavior,
            )
        vec = ak.pad_none(vec, 3)
        vec["norm3"] = np.sqrt(vec.dot(vec))
        vec["idx"] = ak.local_index(vec)
        i, j = ak.unzip(ak.combinations(vec, 2))
        best = ak.argmax((i + j).mass, axis=1, keepdims=True)
        leg1, leg2 = ak.firsts(i[best]), ak.firsts(j[best])
        #assert ak.all((leg1 + leg2).mass == ak.max((i + j).mass, axis=1))
        #leg3 = vec[(best == 0)*2 + (best == 1)*1 + (best == 2)*0]
        leg3 = ak.firsts(vec[(vec.idx != leg1.idx) & (vec.idx != leg2.idx)])
        #assert ak.all(leg3.x != leg1.x)
        #assert ak.all(leg3.x != leg2.x)
        a12 = np.arccos(leg1.dot(leg2) / (leg1.norm3 * leg2.norm3))
        a13 = np.arccos(leg1.dot(leg3) / (leg1.norm3 * leg3.norm3))
        a23 = np.arccos(leg2.dot(leg3) / (leg2.norm3 * leg3.norm3))
        color_ring = ((a13**2 + a23**2)/(a12**2))
        return color_ring

    def d2_calc(fatjet):
        jetdef = fastjet.JetDefinition(fastjet.cambridge_algorithm, 0.8) # make this C/A at 0.8
        pf = ak.flatten(fatjet.constituents.pf, axis=1)
        cluster = fastjet.ClusterSequence(pf, jetdef)
        softdrop = cluster.exclusive_jets_softdrop_grooming()
        softdrop_cluster = fastjet.ClusterSequence(softdrop.constituents, jetdef)
        d2 = softdrop_cluster.exclusive_jets_energy_correlator(func='D2')
        return d2

    class MyProcessor_Signal(processor.ProcessorABC):
        
        def __init__(self):
            pass
    
        
        def process(self, events):
            dataset = events.metadata['dataset']
            
            fatjet = events.FatJet
            
            genhiggs = (events.GenPart[
                (events.GenPart.pdgId==25)
                & events.GenPart.hasFlags(["fromHardProcess", "isLastCopy"])
            ])
            parents = events.FatJet.nearest(genhiggs, threshold=0.1)
            higgs_jets = ~ak.is_none(parents, axis=1)
            
            
            cut = ((fatjet.pt > 300) & (fatjet.msoftdrop > 110) & 
                   (fatjet.msoftdrop < 140) & (abs(fatjet.eta) < 2.5)) & (higgs_jets) & (fatjet.btagDDBvLV2 > 0.65)
            
            boosted_fatjet = fatjet[cut]
            
            uf_cr = ak.unflatten(color_ring(boosted_fatjet), counts=ak.num(boosted_fatjet))
            d2 = ak.unflatten(d2_calc(boosted_fatjet), counts=ak.num(boosted_fatjet))
            boosted_fatjet['color_ring'] = uf_cr
            boosted_fatjet['d2b1'] = d2
            
            hcr = (
                dhist.Hist.new
                .Reg(40, 0, 10, name='color_ring', label='Color_Ring')
                .Weight()
            )
    
            d2b1 = (
                dhist.Hist.new
                .Reg(40, 0, 3, name='D2B1', label='D2B1')
                .Weight()
            )
            
            cmssw_n2 = (
                dhist.Hist.new
                .Reg(40, 0, 0.5, name='cmssw_n2', label='CMSSW_N2')
                .Weight()
            )
            
            cmssw_n3 = (
                dhist.Hist.new
                .Reg(40, 0, 3, name='cmssw_n3', label='CMSSW_N3')
                .Weight()
            )
            
            ncons = (
                dhist.Hist.new
                .Reg(40, 0, 200, name='constituents', label='nConstituents')
                .Weight()
            )
            
            mass = (
                dhist.Hist.new
                .Reg(40, 0, 250, name='mass', label='Mass')
                .Weight()
            )
            
            sdmass = (
                dhist.Hist.new
                .Reg(40, 0, 250, name='sdmass', label='SDmass')
                .Weight()
            )
    
            btag = (
                dhist.Hist.new
                .Reg(40, 0, 1, name='Btag', label='Btag')
                .Weight()
            )
            
            fill_cr = ak.fill_none(ak.flatten(boosted_fatjet.color_ring), 0)
            hcr.fill(color_ring=fill_cr)
            d2b1.fill(D2B1=ak.flatten(boosted_fatjet.d2b1))
            cmssw_n2.fill(cmssw_n2=ak.flatten(boosted_fatjet.n2b1))
            cmssw_n3.fill(cmssw_n3=ak.flatten(boosted_fatjet.n3b1))
            ncons.fill(constituents=ak.flatten(boosted_fatjet.nConstituents))
            mass.fill(mass=ak.flatten(boosted_fatjet.mass))
            sdmass.fill(sdmass=ak.flatten(boosted_fatjet.msoftdrop))
            btag.fill(Btag=ak.flatten(boosted_fatjet.btagDDBvLV2))
            
            return {
                dataset: {
                    "entries": ak.count(events.event, axis=None),
                    "Color_Ring": hcr,
                    "N2": cmssw_n2,
                    "N3": cmssw_n3,
                    "nConstituents": ncons,
                    "Mass": mass,
                    "SDmass": sdmass,
                    "Btag": btag,
                    "D2": d2b1,
                }
            }
        
        def postprocess(self, accumulator):
            pass

    class MyProcessor_Background(processor.ProcessorABC):
        
        def __init__(self):
            pass
    
        
        def process(self, events):
            dataset = events.metadata['dataset']
            
            fatjet = events.FatJet 
            
            cut = ((fatjet.pt > 300) & (fatjet.msoftdrop > 110) & 
                   (fatjet.msoftdrop < 140) & (abs(fatjet.eta) < 2.5))  & (fatjet.btagDDBvLV2 > 0.65)
            
            boosted_fatjet = fatjet[cut]
            
            uf_cr = ak.unflatten(color_ring(boosted_fatjet), counts=ak.num(boosted_fatjet))
            d2 = ak.unflatten(d2_calc(boosted_fatjet), counts=ak.num(boosted_fatjet))
            boosted_fatjet['color_ring'] = uf_cr
            boosted_fatjet['d2b1'] = d2
            
            hcr = (
                dhist.Hist.new
                .Reg(40, 0, 10, name='color_ring', label='Color_Ring')
                .Weight()
            )
    
            d2b1 = (
                dhist.Hist.new
                .Reg(40, 0, 3, name='D2B1', label='D2B1')
                .Weight()
            )
            
            cmssw_n2 = (
                dhist.Hist.new
                .Reg(40, 0, 0.5, name='cmssw_n2', label='CMSSW_N2')
                .Weight()
            )
            
            cmssw_n3 = (
                dhist.Hist.new
                .Reg(40, 0, 3, name='cmssw_n3', label='CMSSW_N3')
                .Weight()
            )
            
            ncons = (
                dhist.Hist.new
                .Reg(40, 0, 200, name='constituents', label='nConstituents')
                .Weight()
            )
            
            mass = (
                dhist.Hist.new
                .Reg(40, 0, 250, name='mass', label='Mass')
                .Weight()
            )
            
            sdmass = (
                dhist.Hist.new
                .Reg(40, 0, 250, name='sdmass', label='SDmass')
                .Weight()
            )
    
            btag = (
                dhist.Hist.new
                .Reg(40, 0, 1, name='Btag', label='Btag')
                .Weight()
            )
            
            fill_cr = ak.fill_none(ak.flatten(boosted_fatjet.color_ring), 0)
            hcr.fill(color_ring=fill_cr)
            d2b1.fill(D2B1=ak.flatten(boosted_fatjet.d2b1))
            cmssw_n2.fill(cmssw_n2=ak.flatten(boosted_fatjet.n2b1))
            cmssw_n3.fill(cmssw_n3=ak.flatten(boosted_fatjet.n3b1))
            ncons.fill(constituents=ak.flatten(boosted_fatjet.nConstituents))
            mass.fill(mass=ak.flatten(boosted_fatjet.mass))
            sdmass.fill(sdmass=ak.flatten(boosted_fatjet.msoftdrop))
            btag.fill(Btag=ak.flatten(boosted_fatjet.btagDDBvLV2))
            
            return {
                dataset: {
                    "entries": ak.count(events.event, axis=None),
                    "Color_Ring": hcr,
                    "N2": cmssw_n2,
                    "N3": cmssw_n3,
                    "nConstituents": ncons,
                    "Mass": mass,
                    "SDmass": sdmass,
                    "Btag": btag,
                    "D2": d2b1,
                }
            }
        
        def postprocess(self, accumulator):
            pass


    start = time.time()
    result = {}
    result["Hgg"] = MyProcessor_Signal().process(events["hgg"])
    print("hbb")
    result["Hbb"] = MyProcessor_Signal().process(events["hbb"])
    print("300")
    result["QCD_Pt_300to470_TuneCP5_13TeV_pythia8"] = MyProcessor_Background().process(
        events["q347"]
    )
    print("470")
    result["QCD_Pt_470to600_TuneCP5_13TeV_pythia8"] = MyProcessor_Background().process(
        events["q476"]
    )
    print("600")
    result["QCD_Pt_600to800_TuneCP5_13TeV_pythia8"] = MyProcessor_Background().process(
        events["q68"]
    )
    print("800")
    result["QCD_Pt_800to1000_TuneCP5_13TeV_pythia8"] = MyProcessor_Background().process(
        events["q810"]
    )
    print("1000")
    result[
        "QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8"
    ] = MyProcessor_Background().process(events["q1014"])
    print("1400")
    result[
        "QCD_Pt_1400to1800_TuneCP5_13TeV_pythia8"
    ] = MyProcessor_Background().process(events["q1418"])
    print("1800")
    result[
        "QCD_Pt_1800to2400_TuneCP5_13TeV_pythia8"
    ] = MyProcessor_Background().process(events["q1824"])
    print("2400")
    result[
        "QCD_Pt_2400to3200_TuneCP5_13TeV_pythia8"
    ] = MyProcessor_Background().process(events["q2432"])
    print("3200")
    result["QCD_Pt_3200toInf_TuneCP5_13TeV_pythia8"] = MyProcessor_Background().process(
        events["q32Inf"]
    )
    stop = time.time()
    print(stop-start)

    print('computing')
    computed = dask.compute(result, scheduler=m.get, resources={"cores": 1}, resources_mode=None, lazy_transfers=True)
    with open('outputs/big_btagged_result.pkl', 'wb') as f:
        pickle.dump(computed, f)


full_stop = time.time()
print('full run time is ' + str((full_stop - full_start)/60))
