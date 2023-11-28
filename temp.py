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

warnings.filterwarnings("ignore", "Found duplicate branch")
warnings.filterwarnings("ignore", "Missing cross-reference index for")
warnings.filterwarnings("ignore", "dcut")
warnings.filterwarnings("ignore", "Please ensure")

q347_files = os.listdir('/project01/ndcms/cmoore24/qcd/300to470')
#q347_files = q347_files[:50]

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

class MyProcessor_Background(processor.ProcessorABC):
    
    def __init__(self):
        pass

    
    def process(self, events):
        # dataset = events.metadata['dataset']
        dataset = '1000to1400'
        
        fatjet = events#.FatJet
        
        cut = ((fatjet.pt > 300) & (fatjet.msoftdrop > 110) & 
               (fatjet.msoftdrop < 140) & (abs(fatjet.eta) < 2.5)) #& (fatjet.btagDDBvLV2 > 0.89)
        
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
            .Reg(40, -1, 1, name='Btag', label='Btag')
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
                #"entries": ak.max(events.event),
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


from ndcctools.taskvine import DaskVine
m = DaskVine([9123,9128], name="jupyter")
q347 = NanoEventsFactory.from_root(
    #[{'/project01/ndcms/cmoore24/qcd/300to470/' + fn: "/Events"} for fn in q347_files],
    {'/project01/ndcms/cmoore24/qcd/300to470/' + fn: "/Events" for fn in q347_files},
    permit_dask=True,
    schemaclass=PFNanoAODSchema,
    metadata={"dataset": "QCD_Pt_300to470"},
).events()

result = MyProcessor_Background().process(q347.FatJet)
print('GO')
compute_q347 = dask.compute(result, scheduler=m.get)
print('I finished :)')

# bad_files = []
# from ndcctools.taskvine import DaskVine
# m = DaskVine([9123,9128], name="jupyter")
# for i in range(0, 349):
#     q1014 = NanoEventsFactory.from_root(
#         {'/project01/ndcms/cmoore24/qcd/300to470/' + q347_files[i]: "/Events"},
#         permit_dask=True,
#         schemaclass=PFNanoAODSchema,
#         metadata={"dataset":"300to470"}
#     ).events()
#     result = MyProcessor_Background().process(q1014.FatJet)
#     print('GO')
#     try:
#         print('computing')
#         dask.compute(result, scheduler=m.get)
#         print('done')
#     except:
#         print('bad')
#         bad_files.append(q347_files[i])
#         continue
#     if i % 20 == 0:
#         print(i)
# print(len(bad_files))
