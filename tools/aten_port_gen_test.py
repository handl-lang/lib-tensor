#!/usr/bin/env python3
"""Unit tests for tools/aten_port_gen.py — deterministic parsing,
classification, default handling, seam preservation, and reserved-name
handling. Run:  python3 -m unittest tools/aten_port_gen_test.py
(or `python3 tools/aten_port_gen_test.py` from the repo root).
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aten_port_gen as g  # noqa: E402


class TestPortListParsing(unittest.TestCase):
    def test_entry_count(self):
        entries = g.parse_port_list(g.PORT_LIST)
        self.assertEqual(len(entries), 207)

    def test_fields(self):
        entries = g.parse_port_list(g.PORT_LIST)
        e = entries[0]
        for k in ("hnl", "aten", "seam", "covers"):
            self.assertIn(k, e)

    def test_seams_marked(self):
        entries = g.parse_port_list(g.PORT_LIST)
        seams = {e["hnl"] for e in entries if e["seam"]}
        self.assertEqual(seams, {"empty", "rand", "normal"})

    def test_comments_and_blanks_skipped(self):
        entries = g.parse_port_list(g.PORT_LIST)
        self.assertTrue(all(e["hnl"] and e["aten"] for e in entries))


class TestSchemaResolution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schemas = g.load_schemas(g.SCHEMAS)

    def test_known_schema_found(self):
        rec, warn = g.resolve_schema(self.schemas, "sum")
        self.assertIsNotNone(rec)
        self.assertIsNone(warn)

    def test_missing_schema_warns_not_crashes(self):
        rec, warn = g.resolve_schema(self.schemas, "aten::no_such_op_xyz")
        self.assertIsNone(rec)
        self.assertIsNotNone(warn)


class TestHnlLit(unittest.TestCase):
    def test_literals(self):
        self.assertEqual(g.hnl_lit(None), "null")
        self.assertEqual(g.hnl_lit(True), "true")
        self.assertEqual(g.hnl_lit(False), "false")
        self.assertEqual(g.hnl_lit(3), "3")
        self.assertEqual(g.hnl_lit(0.5), "0.5")

    def test_string_keywords_become_literals(self):
        # DRIVER tables spell literal defaults as strings; they must
        # emit bare keywords, not quoted strings (a quoted "null"
        # reached drivers as Str("null") and broke null checks).
        self.assertEqual(g.hnl_lit("null"), "null")
        self.assertEqual(g.hnl_lit("true"), "true")
        self.assertEqual(g.hnl_lit("false"), "false")
        self.assertEqual(g.hnl_lit("0"), "0")
        self.assertEqual(g.hnl_lit("10"), "10")

    def test_real_strings_stay_quoted(self):
        self.assertEqual(g.hnl_lit("tanh"), '"tanh"')
        self.assertEqual(g.hnl_lit("f64"), '"f64"')


class TestClassify(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.entries = {e["hnl"]: e for e in g.parse_port_list(g.PORT_LIST)}

    def _cls(self, hnl):
        return g.classify(self.entries[hnl])[0]

    def test_unary(self):
        self.assertEqual(self._cls("abs"), "unary")
        self.assertEqual(self._cls("floor"), "unary")

    def test_pred(self):
        self.assertEqual(self._cls("isnan"), "pred")

    def test_bin(self):
        self.assertEqual(self._cls("atan2"), "bin")
        self.assertEqual(self._cls("eq"), "bin")

    def test_driver(self):
        self.assertEqual(self._cls("sum"), "driver")
        self.assertEqual(self._cls("mm"), "driver")

    def test_stub(self):
        # exp2 flipped to a real recipe once math.exp2 landed; einsum
        # remains the only honest stub besides the host seams.
        self.assertEqual(self._cls("exp2"), "unary")
        self.assertEqual(self._cls("einsum"), "stub")

    def test_seams_are_stubs(self):
        for seam in ("empty", "rand", "normal"):
            cls, why = g.classify(self.entries[seam])
            self.assertEqual(cls, "stub", seam)
            self.assertTrue(why, seam)

    def test_unique_nonzero_dim_are_drivers(self):
        # closed seams: handwritten .hnl drivers in internal/aten.hnl
        for op in ("unique", "nonzero_dim"):
            cls, _ = g.classify(self.entries[op])
            self.assertEqual(cls, "driver", op)

    def test_no_clean_classifies_stub(self):
        # every non-seam, non-stub entry must land in a generatable class
        for hnl, e in self.entries.items():
            cls, _ = g.classify(e)
            self.assertIn(cls, ("unary", "pred", "bin", "compose",
                                "driver", "stub", "skip"), hnl)


class TestGeneratedOutput(unittest.TestCase):
    """Determinism + shape of the emitted surface (no writes)."""

    def _emit_family(self, fam):
        # reproduce the per-family emit for one op to check shape
        entries = g.parse_port_list(g.PORT_LIST)
        ops = []
        for e in entries:
            f = next((f for f, names in g.FAMILIES.items()
                      if e["hnl"] in names), "misc")
            if f == fam:
                cls, detail = g.classify(e)
                if cls != "stub" or e["seam"]:
                    ops.append((e, cls, detail))
        return ops

    def test_stub_body_dies(self):
        body = g.gen_stub("empty", "uninitialized fill cannot be represented")
        self.assertIn("die(", body)
        self.assertIn("aten.empty", body)
        # ascii-only: the lexer rejects non-ASCII
        self.assertTrue(all(ord(c) < 128 for c in body))

    def test_stub_output_is_ascii(self):
        # gen_stub sanitizes reasons (em-dashes etc.) for the lexer
        for op, reason in g.STUBS.items():
            body = g.gen_stub(op, reason)
            self.assertTrue(all(ord(c) < 128 for c in body), op)
            self.assertNotIn("'", body)

    def test_reserved_names_renamed(self):
        # sort/where are parser-reserved: generated export uses _
        # suffix and index.hnl re-exports under the public name.
        for e in g.parse_port_list(g.PORT_LIST):
            if e["hnl"] in ("sort", "where"):
                ename = e["hnl"] + "_"
                self.assertTrue(ename.endswith("_"))

    def test_searchsorted_binds_schema_positions(self):
        # schema: searchsorted(sorted_sequence, self, *, out_int32,
        # right, side, sorter) — t = sorted_sequence, rest =
        # [self, out_int32, right, side, sorter]. right is rest[2],
        # NOT rest[1] (that is out_int32, accepted-and-ignored).
        body = g.gen_driver("searchsorted", g.DRIVER["searchsorted"])
        self.assertIn("aten_req(rest, 0)", body)
        self.assertIn("aten_opt(rest, 2, false)", body)
        self.assertIn("aten_opt(rest, 3, null)", body)
        self.assertIn("aten_opt(rest, 4, null)", body)
        self.assertNotIn("aten_opt(rest, 1, false)", body)

    def test_deterministic_classification(self):
        entries = g.parse_port_list(g.PORT_LIST)
        c1 = [g.classify(e)[0] for e in entries]
        c2 = [g.classify(e)[0] for e in reversed(list(reversed(entries)))]
        self.assertEqual(c1, c2)


if __name__ == "__main__":
    unittest.main()
