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
            "VB":{
                "V1":[
                    ["WPS", 29, 1, 5,6 ],
                    ["HSN2", 18, 1, 2, 6],
                    ["HSN1", 11, 2, 4, 6],
                    ["HSN2", 18, 2, 6, 6],
                    ["WPS", 29, 3, 5, 6],
                    ["HSN1", 11, 3, 5, 6],
                    ["FLS", 4, 4, 6, 6],
                    ["IMP", 20, 4, 5, 6],
                    ["FLS", 4, 5, 6, 6],
                    ["HSN2", 18, 5, 6, 6],
                    ["WPS", 29, 5, 6, 6],
                ],
                "V2":[
                    ["WPS", 29, 1, 3, 6],
                    ["HSN2", 18, 1, 3, 6],
                    ["HSN1", 11, 2, 5, 6],
                    ["WPS", 18, 3, 6, 6],
                    ["FLS", 4, 4, 5, 6],
                    ["IMP", 20, 4, 5, 6],
                    ["FLS", 4, 5, 6, 6],
                    ["HSN2", 18, 5, 6, 6],
                    ["WPS", 46, 5, 6, 6],
                ]
            },
            "HC":{
                "V1":[
                    ["HSN1", 22, 1, 4, 5],
                    ["FLS", 9, 1, 3, 5],
                    ["WPS", 29, 2, 5, 5],
                    ["HSN2", 18, 2, 3, 5],
                    ["WPS", 29, 3, 4, 5],
                    ["HSN1", 22, 3, 4, 5],
                    ["HSN2", 18, 4, 5, 5],
                    ["IMP", 13, 4, 5, 5],
                ],
                "V2":[
                    ["HSN1", 22, 1, 2, 5],
                    ["FLS", 9, 1, 3, 5],
                    ["WPS", 29, 2, 3, 5],
                    ["HSN2", 6, 2, 4, 5],
                    ["WPS", 29, 3, 5, 5],
                    ["HSN2", 18, 3, 4, 5],
                    ["HSN1", 22, 3, 4, 5],
                    ["IMP", 13, 4, 5, 5],
                    ["WPS", 34, 4, 5, 5],
                ]
            }

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
            "vgeo1":67,
            "vgeo2":68,
            "vgeo3":69,
            "vgeo4":74,
            "vgeo5":83,
            "vgeo6":84,
            "hgeo1":104,
            "hgeo2":9,
            "hgeo3":4,
            "hgeo4":49,
            "hgeo5":54,
            "puffer": [95,96,97,98,99],
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
        family = part_obs["variant"][:2].upper()
        all_respots = self.setup[family][variant]
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





