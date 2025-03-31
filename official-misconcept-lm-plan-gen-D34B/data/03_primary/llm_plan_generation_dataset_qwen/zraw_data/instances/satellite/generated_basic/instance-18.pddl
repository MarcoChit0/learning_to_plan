(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	satellite2 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	thermograph1 - mode
	image0 - mode
	star0 - direction
	phenomenon1 - direction
	phenomenon2 - direction
	phenomenon3 - direction
	star4 - direction
	planet5 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 star0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet5)
	(supports instrument1 image0)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 star0)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument2 image0)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 star0)
	(supports instrument3 image0)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 star0)
	(supports instrument4 image0)
	(calibration_target instrument4 star0)
	(supports instrument5 image0)
	(calibration_target instrument5 star0)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 star0)
	(supports instrument7 thermograph1)
	(supports instrument7 image0)
	(calibration_target instrument7 star0)
	(on_board instrument2 satellite2)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon2)
)
(:goal (and
	(pointing satellite0 phenomenon2)
	(pointing satellite1 phenomenon2)
	(have_image phenomenon1 thermograph1)
	(have_image phenomenon2 image0)
	(have_image phenomenon3 image0)
	(have_image star4 thermograph1)
	(have_image planet5 thermograph1)
))

)
