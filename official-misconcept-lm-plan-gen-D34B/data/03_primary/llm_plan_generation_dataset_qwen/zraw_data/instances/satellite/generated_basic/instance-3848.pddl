(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite1 - satellite
	instrument6 - instrument
	instrument7 - instrument
	image2 - mode
	image1 - mode
	thermograph0 - mode
	star0 - direction
	phenomenon1 - direction
	planet2 - direction
	planet3 - direction
	phenomenon4 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 image2)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star0)
	(supports instrument1 image2)
	(calibration_target instrument1 star0)
	(supports instrument2 thermograph0)
	(supports instrument2 image2)
	(supports instrument2 image1)
	(calibration_target instrument2 star0)
	(supports instrument3 image2)
	(calibration_target instrument3 star0)
	(supports instrument4 image1)
	(supports instrument4 image2)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 star0)
	(supports instrument5 image1)
	(calibration_target instrument5 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon4)
	(supports instrument6 image1)
	(supports instrument6 image2)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 star0)
	(supports instrument7 thermograph0)
	(supports instrument7 image2)
	(supports instrument7 image1)
	(calibration_target instrument7 star0)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet3)
)
(:goal (and
	(have_image phenomenon1 image1)
	(have_image planet2 image1)
	(have_image planet3 thermograph0)
	(have_image phenomenon4 thermograph0)
))

)
