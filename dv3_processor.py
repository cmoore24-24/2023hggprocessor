from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema
import fastjet
import numpy as np
import awkward as ak
from coffea import processor
import coffea.nanoevents.methods.vector as vector
import warnings
import hist.dask as dhist
import dask
import pickle
import os
from ndcctools.taskvine import DaskVine
import time

full_start = time.time()

if __name__ == "__main__":
    m = DaskVine(
        [9123, 9128],
        name=f"{os.environ['USER']}-hgg",
        run_info_path=f"/project01/ndcms/{os.environ['USER']}/vine-run-info",
    )

    available_functions = [
        "Color_Ring",
        "Color_Ring_Var",
        "D2",
        "D3",
        "N4",
        "U1",
        "U2",
        "U3",
        "MRatio",
        "N2",
        "N3",
        "nConstituents",
        "Mass",
        "SDmass",
        "Btag",
    ]

    enabled_functions = set()
    enabled_functions.add("Btag")

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
            info["files"] = os.listdir(f"{path_root}/{info['path']}")

        with open("data_dv2.pkl", "wb") as fw:
            pickle.dump(datasets, fw)

    source = "/project01/ndcms/cmoore24"
    events = {}
    for name, info in datasets.items():
        events[name] = NanoEventsFactory.from_root(
            {f"{source}/{info['path']}/{fn}": "/Events" for fn in info["files"]},
            # permit_dask=True,
            schemaclass=PFNanoAODSchema,
            # uproot_options={"chunks_per_file":1},
            metadata={"dataset": info["label"]},
        ).events()

    def color_ring(fatjet, variant=False):
        pf = ak.flatten(fatjet.constituents.pf, axis=1)
        jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.2)
        cluster = fastjet.ClusterSequence(pf, jetdef)
        subjets = cluster.inclusive_jets()
        vec = ak.zip(
            {
                "x": subjets.px,
                "y": subjets.py,
                "z": subjets.pz,
                "t": subjets.E,
            },
            with_name="LorentzVector",
            behavior=vector.behavior,
        )
        vec = ak.pad_none(vec, 3)
        vec["norm3"] = np.sqrt(vec.dot(vec))
        vec["idx"] = ak.local_index(vec)
        i, j, k = ak.unzip(ak.combinations(vec, 3))
        best = ak.argmin(abs((i + j + k).mass - 125), axis=1, keepdims=True)
        order_check = ak.concatenate([i[best].mass, j[best].mass, k[best].mass], axis=1)
        largest = ak.argmax(order_check, axis=1, keepdims=True)
        smallest = ak.argmin(order_check, axis=1, keepdims=True)
        leading_particles = ak.concatenate([i[best], j[best], k[best]], axis=1)
        leg1 = leading_particles[largest]
        leg3 = leading_particles[smallest]
        leg2 = leading_particles[
            (leading_particles.idx != ak.flatten(leg1.idx))
            & (leading_particles.idx != ak.flatten(leg3.idx))
        ]
        leg1 = ak.firsts(leg1)
        leg2 = ak.firsts(leg2)
        leg3 = ak.firsts(leg3)
        a12 = np.arccos(leg1.dot(leg2) / (leg1.norm3 * leg2.norm3))
        a13 = np.arccos(leg1.dot(leg3) / (leg1.norm3 * leg3.norm3))
        a23 = np.arccos(leg2.dot(leg3) / (leg2.norm3 * leg3.norm3))
        if not variant:
            color_ring = (a13**2 + a23**2) / (a12**2)
        else:
            color_ring = a13**2 + a23**2 - a12**2
        return color_ring

    def d2_calc(fatjet):
        jetdef = fastjet.JetDefinition(
            fastjet.cambridge_algorithm, 0.8
        )  # make this C/A at 0.8
        pf = ak.flatten(fatjet.constituents.pf, axis=1)
        cluster = fastjet.ClusterSequence(pf, jetdef)
        softdrop = cluster.exclusive_jets_softdrop_grooming()
        softdrop_cluster = fastjet.ClusterSequence(softdrop.constituents, jetdef)
        d2 = softdrop_cluster.exclusive_jets_energy_correlator(func="D2")
        return d2

    def n4_calc(fatjet):
        jetdef = fastjet.JetDefinition(
            fastjet.cambridge_algorithm, 0.8
        )  # make this C/A at 0.8
        pf = ak.flatten(fatjet.constituents.pf, axis=1)
        cluster = fastjet.ClusterSequence(pf, jetdef)
        softdrop = cluster.exclusive_jets_softdrop_grooming()
        softdrop_cluster = fastjet.ClusterSequence(softdrop.constituents, jetdef)
        e25 = softdrop_cluster.exclusive_jets_energy_correlator(
            func="generalized", angles=2, npoint=5
        )
        e14 = softdrop_cluster.exclusive_jets_energy_correlator(
            func="generalized", angles=1, npoint=4
        )
        n4 = e25 / (e14**2)
        return n4

    def d3_calc(fatjet):
        jetdef = fastjet.JetDefinition(
            fastjet.cambridge_algorithm, 0.8
        )  # make this C/A at 0.8
        pf = ak.flatten(fatjet.constituents.pf, axis=1)
        cluster = fastjet.ClusterSequence(pf, jetdef)
        softdrop = cluster.exclusive_jets_softdrop_grooming()
        softdrop_cluster = fastjet.ClusterSequence(softdrop.constituents, jetdef)
        e4 = softdrop_cluster.exclusive_jets_energy_correlator(
            func="generic", normalized=True, npoint=4
        )
        e2 = softdrop_cluster.exclusive_jets_energy_correlator(
            func="generic", normalized=True, npoint=2
        )
        e3 = softdrop_cluster.exclusive_jets_energy_correlator(
            func="generic", normalized=True, npoint=3
        )
        d3 = (e4 * (e2**3)) / (e3**3)
        return d3

    def u_calc(fatjet, n):
        jetdef = fastjet.JetDefinition(
            fastjet.cambridge_algorithm, 0.8
        )  # make this C/A at 0.8
        pf = ak.flatten(fatjet.constituents.pf, axis=1)
        cluster = fastjet.ClusterSequence(pf, jetdef)
        softdrop = cluster.exclusive_jets_softdrop_grooming()
        softdrop_cluster = fastjet.ClusterSequence(softdrop.constituents, jetdef)
        if n == 1:
            u = softdrop_cluster.exclusive_jets_energy_correlator(func="u1")
        if n == 2:
            u = softdrop_cluster.exclusive_jets_energy_correlator(func="u2")
        if n == 3:
            u = softdrop_cluster.exclusive_jets_energy_correlator(func="u3")
        return u

    class MyProcessor(processor.ProcessorABC):
        def __init__(self):
            pass

        def process(self, events):
            dataset = events.metadata["dataset"]
            computations = {"entries": ak.count(events.event, axis=None)}

            fatjet = events.FatJet

            if "QCD" in dataset:
                print("background")
                cut = (
                    (fatjet.pt > 300)
                    & (fatjet.msoftdrop > 110)
                    & (fatjet.msoftdrop < 140)
                    & (abs(fatjet.eta) < 2.5)
                )  # & (fatjet.btagDDBvLV2 > 0.20)

            else:
                print("signal")
                genhiggs = events.GenPart[
                    (events.GenPart.pdgId == 25)
                    & events.GenPart.hasFlags(["fromHardProcess", "isLastCopy"])
                ]
                parents = events.FatJet.nearest(genhiggs, threshold=0.1)
                higgs_jets = ~ak.is_none(parents, axis=1)

                cut = (
                    (fatjet.pt > 300)
                    & (fatjet.msoftdrop > 110)
                    & (fatjet.msoftdrop < 140)
                    & (abs(fatjet.eta) < 2.5)
                ) & (
                    higgs_jets
                )  # & (fatjet.btagDDBvLV2 > 0.20)

            boosted_fatjet = fatjet[cut]
            boosted_fatjet.constituents.pf["pt"] = (
                boosted_fatjet.constituents.pf.pt
                * boosted_fatjet.constituents.pf.puppiWeight
            )

            if "Color_Ring" in enabled_functions:
                uf_cr = ak.unflatten(
                    color_ring(boosted_fatjet), counts=ak.num(boosted_fatjet)
                )
                boosted_fatjet["color_ring"] = uf_cr
                hcr = dhist.Hist.new.Reg(
                    100, 0.5, 4.5, name="color_ring", label="Color_Ring"
                ).Weight()
                fill_cr = ak.fill_none(ak.flatten(boosted_fatjet.color_ring), 0)
                hcr.fill(color_ring=fill_cr)
                computations["Color_Ring"] = hcr

            if "Color_Ring_Var" in enabled_functions:
                uf_cr_var = ak.unflatten(
                    color_ring(boosted_fatjet, variant=True),
                    counts=ak.num(boosted_fatjet),
                )
                boosted_fatjet["color_ring_var"] = uf_cr_var
                hcr_var = dhist.Hist.new.Reg(
                    40, 0, 3, name="color_ring_var", label="Color_Ring_Var"
                ).Weight()
                fill_cr_var = ak.fill_none(ak.flatten(boosted_fatjet.color_ring_var), 0)
                hcr_var.fill(color_ring_var=fill_cr_var)
                computations["Color_Ring_Var"] = hcr_var

            if "D2" in enabled_functions:
                d2 = ak.unflatten(
                    d2_calc(boosted_fatjet), counts=ak.num(boosted_fatjet)
                )
                boosted_fatjet["d2b1"] = d2
                d2b1 = dhist.Hist.new.Reg(40, 0, 3, name="D2B1", label="D2B1").Weight()
                d2b1.fill(D2B1=ak.flatten(boosted_fatjet.d2b1))
                computations["D2"] = d2b1

            if "D3" in enabled_functions:
                d3 = ak.unflatten(
                    d3_calc(boosted_fatjet), counts=ak.num(boosted_fatjet)
                )
                boosted_fatjet["d3b1"] = d3
                d3b1 = dhist.Hist.new.Reg(40, 0, 3, name="D3B1", label="D3B1").Weight()
                d3b1.fill(D3B1=ak.flatten(boosted_fatjet.d3b1))
                computations["D3"] = d3b1

            if "N4" in enabled_functions:
                n4 = ak.unflatten(
                    n4_calc(boosted_fatjet), counts=ak.num(boosted_fatjet)
                )
                boosted_fatjet["n4b1"] = n4
                n4b1 = dhist.Hist.new.Reg(40, 0, 35, name="N4B1", label="N4B1").Weight()
                n4b1.fill(N4B1=ak.flatten(boosted_fatjet.n4b1))
                computations["N4"] = n4b1

            if "U1" in enabled_functions:
                u1 = ak.unflatten(
                    u_calc(boosted_fatjet, n=1), counts=ak.num(boosted_fatjet)
                )
                boosted_fatjet["u1b1"] = u1
                u1b1 = dhist.Hist.new.Reg(
                    40, 0, 0.3, name="U1B1", label="U1B1"
                ).Weight()
                u1b1.fill(U1B1=ak.flatten(boosted_fatjet.u1b1))
                computations["U1"] = u1b1

            if "U2" in enabled_functions:
                u2 = ak.unflatten(
                    u_calc(boosted_fatjet, n=2), counts=ak.num(boosted_fatjet)
                )
                boosted_fatjet["u2b1"] = u2
                u2b1 = dhist.Hist.new.Reg(
                    40, 0, 0.05, name="U2B1", label="U2B1"
                ).Weight()
                u2b1.fill(U2B1=ak.flatten(boosted_fatjet.u2b1))
                computations["U2"] = u2b1

            if "U3" in enabled_functions:
                u3 = ak.unflatten(
                    u_calc(boosted_fatjet, n=3), counts=ak.num(boosted_fatjet)
                )
                boosted_fatjet["u3b1"] = u3
                u3b1 = dhist.Hist.new.Reg(
                    40, 0, 0.05, name="U3B1", label="U3B1"
                ).Weight()
                u3b1.fill(U3B1=ak.flatten(boosted_fatjet.u3b1))
                computations["U3"] = u3b1

            if "MRatio" in enabled_functions:
                mass_ratio = boosted_fatjet.mass / boosted_fatjet.msoftdrop
                boosted_fatjet["mass_ratio"] = mass_ratio
                mosm = dhist.Hist.new.Reg(
                    40, 0.9, 1.5, name="MRatio", label="MRatio"
                ).Weight()
                mosm.fill(MRatio=ak.flatten(boosted_fatjet.mass_ratio))
                computations["MRatio"] = mosm

            if "N2" in enabled_functions:
                cmssw_n2 = dhist.Hist.new.Reg(
                    40, 0, 0.5, name="cmssw_n2", label="CMSSW_N2"
                ).Weight()
                cmssw_n2.fill(cmssw_n2=ak.flatten(boosted_fatjet.n2b1))
                computations["N2"] = cmssw_n2

            if "N3" in enabled_functions:
                cmssw_n3 = dhist.Hist.new.Reg(
                    40, 0, 3, name="cmssw_n3", label="CMSSW_N3"
                ).Weight()
                cmssw_n3.fill(cmssw_n3=ak.flatten(boosted_fatjet.n3b1))
                computations["N3"] = cmssw_n3

            if "nConstituents" in enabled_functions:
                ncons = dhist.Hist.new.Reg(
                    40, 0, 200, name="constituents", label="nConstituents"
                ).Weight()
                ncons.fill(constituents=ak.flatten(boosted_fatjet.nConstituents))
                computations["nConstituents"] = ncons

            if "Mass" in enabled_functions:
                mass = dhist.Hist.new.Reg(
                    40, 0, 250, name="mass", label="Mass"
                ).Weight()
                mass.fill(mass=ak.flatten(boosted_fatjet.mass))
                computations["Mass"] = mass

            if "SDmass" in enabled_functions:
                sdmass = dhist.Hist.new.Reg(
                    40, 0, 250, name="sdmass", label="SDmass"
                ).Weight()
                sdmass.fill(sdmass=ak.flatten(boosted_fatjet.msoftdrop))
                computations["SDmass"] = sdmass

            if "Btag" in enabled_functions:
                btag = dhist.Hist.new.Reg(40, 0, 1, name="Btag", label="Btag").Weight()
                btag.fill(Btag=ak.flatten(boosted_fatjet.btagDDBvLV2))
                computations["Btag"] = btag

            return {dataset: computations}

        def postprocess(self, accumulator):
            pass

    start = time.time()
    result = {}
    result["Hgg"] = MyProcessor().process(events["hgg"])
    print("hbb")
    result["Hbb"] = MyProcessor().process(events["hbb"])
    print("300")
    result["QCD_Pt_300to470_TuneCP5_13TeV_pythia8"] = MyProcessor().process(
        events["q347"]
    )
    print("470")
    result["QCD_Pt_470to600_TuneCP5_13TeV_pythia8"] = MyProcessor().process(
        events["q476"]
    )
    print("600")
    result["QCD_Pt_600to800_TuneCP5_13TeV_pythia8"] = MyProcessor().process(
        events["q68"]
    )
    print("800")
    result["QCD_Pt_800to1000_TuneCP5_13TeV_pythia8"] = MyProcessor().process(
        events["q810"]
    )
    print("1000")
    result["QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8"] = MyProcessor().process(
        events["q1014"]
    )
    print("1400")
    result["QCD_Pt_1400to1800_TuneCP5_13TeV_pythia8"] = MyProcessor().process(
        events["q1418"]
    )
    print("1800")
    result["QCD_Pt_1800to2400_TuneCP5_13TeV_pythia8"] = MyProcessor().process(
        events["q1824"]
    )
    print("2400")
    result["QCD_Pt_2400to3200_TuneCP5_13TeV_pythia8"] = MyProcessor().process(
        events["q2432"]
    )
    print("3200")
    result["QCD_Pt_3200toInf_TuneCP5_13TeV_pythia8"] = MyProcessor().process(
        events["q32Inf"]
    )
    stop = time.time()
    print(stop - start)

    print("computing")
    computed = dask.compute(
        result,
        scheduler=m.get,
        resources={"cores": 1},
        resources_mode=None,
        lazy_transfers=True,
    )
    with open("../outputs/color_ring_and_u1.pkl", "wb") as f:
        pickle.dump(computed, f)


full_stop = time.time()
print("full run time is " + str((full_stop - full_start) / 60))
