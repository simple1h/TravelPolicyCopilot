CREATE TABLE IF NOT EXISTS employees (
    employee_id TEXT PRIMARY KEY,
    employee_category TEXT NOT NULL,
    cost_center TEXT NOT NULL,
    grade TEXT NOT NULL,
    home_city TEXT NOT NULL,
    policy_tier TEXT NOT NULL,
    max_hotel_per_night INT NOT NULL,
    max_flight_class TEXT NOT NULL,
    requires_preapproval_over_amount INT NOT NULL,
    accessibility_requirements JSONB NOT NULL DEFAULT '[]'::jsonb
);

CREATE TABLE IF NOT EXISTS approval_requests (
    request_id TEXT PRIMARY KEY,
    employee_id TEXT NOT NULL REFERENCES employees(employee_id),
    trip_request JSONB NOT NULL,
    selected_option_id TEXT NOT NULL,
    total_price INT NOT NULL,
    budget_result JSONB NOT NULL,
    policy_results JSONB NOT NULL,
    citations JSONB NOT NULL,
    status TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    idempotency_key TEXT NOT NULL UNIQUE
);

CREATE INDEX IF NOT EXISTS idx_approval_requests_employee_created
ON approval_requests (employee_id, created_at DESC);
