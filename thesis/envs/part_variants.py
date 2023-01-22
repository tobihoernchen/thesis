class PartVariant:
    def __init__(self) -> None:
        pass


class MatrixPart(PartVariant):
    def __init__(self) -> None:
        super().__init__()
        #Verfahren, Anzahl, nach, vor, NacharbeitVor
        self.max_procedure = 100
        self.n_geos = 6
        self.setup = {
            "V1":[
                ["WPS", 20, 1, 3,3 ],
                ["WPS", 17, 2, 5, 5],
                ["WPS", 10, 4, 5, 5],
                ["WPS", 15, 3, 4, 4],
                ["HSN1", 18, 1, 5, 5],
                ["HSN1", 7, 2, 3, 3],
                ["HSN2", 20, 3, 5, 5],
                ["FLS", 12, 4, 5, 5],
                ["FLS", 16, 2, 3, 3],
                ["IMP", 50, 2, 5, 5],
            ],
            "V2":[
                ["WPS", 20, 1, 3,3 ],
                ["WPS", 17, 2, 5, 5],
                ["WPS", 10, 4, 5, 5],
                ["WPS", 15, 3, 4, 4],
                ["HSN1", 18, 1, 5, 5],
                ["HSN2", 7, 2, 3, 3],
                ["HSN2", 20, 3, 5, 5],
                ["FLS", 12, 4, 5, 5],
                ["FLS", 6, 2, 3, 3],
                ["IMP", 50, 2, 5, 5],
            ]
        }

        self.procedures = [
            "HSN1",
            "HSN2",
            "WPS",
            "FLS",
            "IMP"
        ]

        self.proc_stat={
            "WPS": ["wps_1","wps_2", "wps_3"],
            "HSN1": ["hsn1_1", "hsn1_2"],
            "HSN2": ["hsn2_1", "hsn2_2"],
            "FLS": ["fls_1", "fls_2"],
            "IMP": ["impact"]
        }

        self.stat_node = {
            "vgeo1":76,
            "vgeo2":77,
            "vgeo3":78,
            "vgeo4":83,
            "vgeo5":92,
            "vgeo6":93,
            "hgeo1":59,
            "hgeo2":9,
            "hgeo3":4,
            "hgeo4":49,
            "hgeo5":54,
            "puffer": [104,105,106,107,108],
            "hsn2_1": 44 ,
            "fls_1": 19 ,
            "hsn2_2":42,
            "hsn1_1": 29,
            "wps_1":40,
            "wps_2":14,
            "wps_3":43,
            "impact":24,
            "hsn1_2":41,
            "fls_2":34,
            "rework":39,
            }

    def translate(self, part_obs):
        before_geo = int(part_obs["state"][-1])
        after_geo = before_geo -1
        variant = "V" + part_obs["variant"][-1]
        all_respots = self.setup[variant]
        open_respots = [respot for respot, done in zip(all_respots, part_obs["respots"]) if not done]
        nio_respots = [respot for respot, nio in zip(all_respots, part_obs["nio"]) if nio]
        possible_respots = [respot for respot in open_respots if respot[2] <= after_geo]
        haveto_respots = [respot for respot in open_respots if respot[3] == before_geo]
        haveto_nio_respots = [respot for respot in nio_respots if respot[4] == before_geo]
        next_geo = before_geo if len(haveto_respots) == 0 and len(haveto_nio_respots) == 0 else None
        respots_per_proc = [sum([respot[1] for respot in possible_respots if respot[0] == proc]) for proc in self.procedures]
        rework = len(nio_respots) > 0
        return {
            "variant": part_obs["variant"],
            "next_geo": next_geo,
            "proc": self.procedures,
            "amount": respots_per_proc,
            "rework": rework
        }





