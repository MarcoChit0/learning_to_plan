(define (problem depot-1-4-2-5-6-3) (:domain depots)
(:objects
	depot0 - Depot
	distributor0 distributor1 distributor2 distributor3 - Distributor
	truck0 truck1 - Truck
	pallet0 pallet1 pallet2 pallet3 pallet4 - Pallet
	crate0 crate1 crate2 - Crate
	hoist0 hoist1 hoist2 hoist3 hoist4 hoist5 - Hoist)
(:init
	(at pallet0 depot0)
	(clear crate0)
	(at pallet1 distributor0)
	(clear crate1)
	(at pallet2 distributor1)
	(clear crate2)
	(at pallet3 distributor2)
	(clear pallet3)
	(at pallet4 distributor3)
	(clear pallet4)
	(at truck0 distributor0)
	(at truck1 distributor3)
	(at hoist0 depot0)
	(available hoist0)
	(at hoist1 distributor0)
	(available hoist1)
	(at hoist2 distributor1)
	(available hoist2)
	(at hoist3 distributor2)
	(available hoist3)
	(at hoist4 distributor3)
	(available hoist4)
	(at hoist5 distributor2)
	(available hoist5)
	(at crate0 depot0)
	(on crate0 pallet0)
	(at crate1 distributor0)
	(on crate1 pallet1)
	(at crate2 distributor1)
	(on crate2 pallet2)
)

(:goal (and
		(on crate0 pallet1)
		(on crate1 pallet4)
		(on crate2 pallet2)
	)
))
