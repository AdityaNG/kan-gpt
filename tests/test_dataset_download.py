import os
from tempfile import TemporaryDirectory
import pytest
import requests_mock

from kan_gpt.download_dataset import download_tinyshakespeare, download_webtext


@pytest.fixture
def mock_api():
    with requests_mock.Mocker() as m:
        yield m


def test_download_tinyshakespeare(mock_api):
    import requests

    base_url = "http://tinyshakespeare/v1"
    response_data = "Out of thy sleep. What is it thou didst say?"

    with TemporaryDirectory() as download_path:
        mock_api.get(f"{base_url}/input.txt", text=response_data)

        assert response_data == requests.get(f"{base_url}/input.txt").text

        download_tinyshakespeare(
            download_path=download_path, base_url=base_url
        )

        file_path = os.path.join(download_path, "input.txt")
        assert os.path.exists(file_path), f"File not found: {file_path}"


def test_download_webtext(mock_api):

    splits = ["test", "train", "valid"]
    base_url = "http://webtext/v1"
    response_data = '{"id": 259999, "ended": true, "length": 488, "text": "This post goes out to those of you who have not yet experienced the pure joy of brewing your morning joe with a Keurig. To those who have wondered just how those little K-cup thingys work and especially to those who have not heard of this new fangled technology at all (you must be living under a rock if you fall in that category!). Keurig brewers are specially designed to extract coffee from a K-cup. A K-cup is a highly engineered, technologically sophisticated mini coffee brewer inside of a very tiny body. K-cups consist of four parts:\n\nThe outer plastic casing. This is a special designed housing that blocks out moisture, light and air allowing your precious coffee to remain perfectly fresh until you are ready to brew it.\n\nA permeable paper filter. The paper filter allows for optimal flavor extraction.\n\nA foil seal. The foil seal keep the coffee air tight and blocks out oxygen and humidity.\n\nK-cups contain a perfectly measured amount of coffee creating a consistent brew cup after cup.\n\nTo brew coffee with a Keurig, the first step is to place the K-cup in the brew chamber of the unit. When the handle is pulled down closing the chamber, a small needle will puncture the foil lid of the K-cup penetrating the coffee with pressurized water at high temperature.The coffee is filtered through the paper. At the same time a second needle punctures the bottom of the K-cup, allowing freshly brewed coffee to pour into your mug.\n\nThis video does a nice job of demonstrating the brewing process:\n\nKeurig K-cup technology is incredibly revolutionary yet incredibly simple at the same time. The creators of the Keurig system shrunk coffee brewing down to its smallest form- one cup at a time. The K-cup is really just a tiny little drip style coffee brewer. The concept is exactly the same however, the technology of the Keurig itself is far more sophisticated.\n\nMany have tried to duplicate this technology but few have come close to matching the quality. So many consumers choose to brew their own coffee grinds with special adapters. Although they seem like a cost saver up front, we have yet to find one that operates as easily and tastes as good as a genuine Kcup.\n\nIf you are reading this article you may also be interested in:"}'  # noqa

    with TemporaryDirectory() as download_path:
        for split in splits:
            mock_api.get(
                f"{base_url}/webtext.{split}.jsonl", text=response_data
            )

        download_webtext(
            download_path=download_path,
            base_url=base_url,
            splits=splits,
        )

        for split in splits:
            file_path = f"{download_path}/webtext.{split}.jsonl"
            assert os.path.exists(file_path), f"File not found: {file_path}"
