from eval.priorknowledge.priorknowledge import PriorKnowledge


def test_generate_service_to_service_routes():
    pk = PriorKnowledge(target_app='train-ticket')

    res = pk.get_service_routes('ts-route-plan')
    assert res == [('ts-travel-plan', 'ts-ui-dashboard'), ('ts-ui-dashboard',)]
    res = pk.get_service_routes('ts-avatar')
    assert res == [('ts-ui-dashboard',)]
