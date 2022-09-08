from meltria.priorknowledge.priorknowledge import new_knowledge


def test_generate_service_to_service_routes():
    pk = new_knowledge("train-ticket", mappings={})

    res = pk.get_service_routes("ts-route-plan")
    assert res == [("ts-travel-plan", "ts-ui-dashboard"), ("ts-ui-dashboard",)]
    res = pk.get_service_routes("ts-avatar")
    assert res == [("ts-ui-dashboard",)]
