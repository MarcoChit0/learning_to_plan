(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	thermograph0 - mode
	star2 - direction
	groundstation3 - direction
	groundstation1 - direction
	star0 - direction
	phenomenon4 - direction
	planet5 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 groundstation3)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star2)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 groundstation3)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 groundstation3)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 star0)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet5)
)
(:goal (and
	(have_image phenomenon4 thermograph0)
	(have_image planet5 thermograph0)
))

)
