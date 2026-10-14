"""G3: clean addresses[], and the location stored as {area, city, country, postcode}, Title Cased."""
import os
import unittest
from unittest import mock

from scraping.locations import clean_addresses, find_addresses_in_text, location_input_problem, parse_location_input, rescue_address

NL = chr(10)


class AddressCleaningTests(unittest.TestCase):
    JUNK = ["14 unique rest", "2007 in an old basement bar in Covent Gard", "75 guests in effortless st",
            "Open daily 9-5. Book now!", "Registered in England No 123456", ""]

    def test_slogans_and_fragments_are_dropped(self):
        self.assertEqual(clean_addresses(self.JUNK), [])

    def test_real_addresses_are_kept_in_order_without_duplicates(self):
        raw = ["4 Derry Street, London, England", "27c Eccleston Place, London SW1W 9NF", "4 derry street, london, england", "123 Main St, Austin, TX 78701"]
        self.assertEqual(clean_addresses(raw), ["4 Derry Street, London, England", "27c Eccleston Place, London SW1W 9NF", "123 Main St, Austin, TX 78701"])

    def test_an_address_glued_to_other_text_is_cut_out(self):
        glued = "2024 All rights reserved ADDRESS 27c Eccleston Place, London SW1W 9NF PHONE +44 20 3468 8751"
        self.assertEqual(clean_addresses([glued]), ["27c Eccleston Place, London, SW1W 9NF"])
        self.assertEqual(rescue_address("Join 5 Soho Street London W1D 3AA today"), "5 Soho Street, London, W1D 3AA")

    def test_an_address_printed_on_three_lines_without_commas_is_found(self):
        page = "(c) 2024 All rights reserved ADDRESS" + NL + "27c Eccleston Place" + NL + "London" + NL + "SW1W 9NF" + NL + "(Located behind Studio Pottery)"
        self.assertEqual(find_addresses_in_text(page), ["27c Eccleston Place, London, SW1W 9NF"])

    def test_page_text_with_no_address_gives_nothing(self):
        self.assertEqual(find_addresses_in_text("We serve 75 guests in effortless style at 14 unique restaurants. Est. 2007."), [])

    def test_street_suffix_letters_inside_a_word_are_not_a_street(self):
        self.assertEqual(find_addresses_in_text("There are 12 Georgian-style rooms. 99 guests in effortless style."), [])


class ParseLocationTests(unittest.TestCase):
    def test_title_case_and_split(self):
        cases = {
            "london, uk": {"area": "", "city": "London", "country": "UK"},
            "belgravia, london": {"area": "Belgravia", "city": "London", "country": ""},
            "Camber, Rye, UK": {"area": "Camber", "city": "Rye", "country": "UK"},
            "MANCHESTER": {"area": "", "city": "Manchester", "country": ""},
            "st albans": {"area": "", "city": "St Albans", "country": ""},
            "Stratford-upon-Avon, England": {"area": "", "city": "Stratford-upon-Avon", "country": "UK"},
            "greater manchester, uk": {"area": "", "city": "Manchester", "country": "UK"},
            "soho, westminster, london": {"area": "Soho, Westminster", "city": "London", "country": ""},
        }
        for text, want in cases.items():
            got = parse_location_input(text)
            self.assertEqual({k: got[k] for k in want}, want, text)

    def test_us_state_is_dropped_and_usa_is_a_country(self):
        got = parse_location_input("new york, ny, usa")
        self.assertEqual((got["city"], got["country"], got["label"]), ("New York", "US", "New York, US"))

    def test_label_is_the_title_cased_canonical_text(self):
        self.assertEqual(parse_location_input("  belgravia ,   london ")["label"], "Belgravia, London")

    def test_postcode_is_kept_as_its_own_field(self):
        got = parse_location_input("London, UK", "sw1w 9nf")
        self.assertEqual((got["city"], got["postcode"]), ("London", "SW1W 9NF"))

    def test_never_raises_on_junk(self):
        for v in (None, "", "   ", 5, ",,,", "SW1W 9NF"):
            self.assertIsInstance(parse_location_input(v), dict)


class LocationProblemTests(unittest.TestCase):
    def test_accepts_a_city_or_area(self):
        for v in ("London, UK", "Belgravia", "Camber, Rye", "New York, NY"):
            self.assertEqual(location_input_problem(v), "", v)

    def test_rejects_non_locations_with_a_reason(self):
        for v in ("", "x", "27c Eccleston Place, London SW1W 9NF", "SW1W 9NF", "90210", "https://x.com", "me@x.com", "1234",
                  "2007 in an old basement bar in Covent Gard", "Delamina Marylebone, 56-58 Marylebone Lane London", "United Kingdom", "UK", "usa"):
            self.assertTrue(location_input_problem(v), v)
        self.assertIn("street address or postcode", location_input_problem("27c Eccleston Place, London SW1W 9NF"))
        self.assertEqual(location_input_problem("a" * 90), "Keep the location short - a city or area is enough.")


os.environ["PHASE5_ALWAYS_RUN_DEEP"] = "false"
try:
    import main
except Exception as exc:  # pragma: no cover
    main = None
    _skip = f"backend app could not be imported here: {exc}"
else:
    _skip = None

if main is not None:
    from bson import ObjectId
    from tests.test_scan_run import FakeCol


@unittest.skipIf(main is None, _skip)
class SavingTests(unittest.IsolatedAsyncioTestCase):
    async def _save(self, location, postcode=None, existing=None):
        biz = FakeCol([existing] if existing else [])
        user = {"id": "u1", "email": "a@b.co"}
        with mock.patch.object(main, "businesses_col", biz):
            await main._upsert_user_business(current_user=user, url="https://x.co.uk", location=location, location_postcode=postcode, business_name="X")
        return biz.docs[0] if biz.docs else None

    async def test_new_location_is_title_cased_and_stored_structured(self):
        doc = await self._save("belgravia, london, uk", postcode="sw1w 9nf")
        self.assertEqual(doc["location"], "Belgravia, London, UK")
        self.assertEqual(doc["location_data"], {"area": "Belgravia", "city": "London", "country": "UK", "postcode": "SW1W 9NF"})

    async def test_an_invalid_location_does_not_replace_the_last_good_one(self):
        existing = {"_id": "1", "user_id": "u1", "normalized_domain": "x.co.uk", "location": "Leeds, UK", "location_data": {"city": "Leeds"}}
        doc = await self._save("27c Eccleston Place, London SW1W 9NF", existing=existing)
        self.assertEqual((doc["location"], doc["location_data"]["city"]), ("Leeds, UK", "Leeds"))

    async def test_public_document_derives_structure_for_older_records_and_flags_junk(self):
        old = {"_id": ObjectId(), "user_id": "u1", "url": "https://x.co.uk", "normalized_domain": "x.co.uk", "location": "Soho, London"}
        pub = main._public_business_doc(old)
        self.assertEqual((pub["locationData"]["area"], pub["locationData"]["city"]), ("Soho", "London"))
        self.assertIsNone(pub["locationProblem"])
        junk = {**old, "_id": ObjectId(), "location": "2007 in an old basement bar in Covent Gard"}
        pub = main._public_business_doc(junk)
        self.assertIsNone(pub["locationData"])
        self.assertTrue(pub["locationProblem"])


if __name__ == "__main__":
    unittest.main()
