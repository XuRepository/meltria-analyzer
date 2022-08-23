from eval.priorknowledge.train_ticket import generate_service_to_service_routes


def test_generate_service_to_service_routes():
    res = generate_service_to_service_routes()
    assert res['ts-avatar'] == [('ts-ui-dashboard',)]
    assert res['ts-route-plan'] == [('ts-travel-plan', 'ts-ui-dashboard'), ('ts-ui-dashboard',)]
