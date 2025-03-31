(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite1 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	image1 - mode
	thermograph0 - mode
	star0 - direction
	groundstation1 - direction
	star2 - direction
	star3 - direction
	star4 - direction
	planet5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 image1)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 image1)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 image1)
	(calibration_target instrument3 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument4 image1)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 image1)
	(calibration_target instrument5 star2)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 star3)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon6)
)
(:goal (and
	(have_image star4 thermograph0)
	(have_image planet5 thermograph0)
	(have_image phenomenon6 image1)
))

)
