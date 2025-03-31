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
	satellite2 - satellite
	instrument5 - instrument
	satellite3 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	thermograph1 - mode
	thermograph3 - mode
	thermograph0 - mode
	image2 - mode
	groundstation2 - direction
	star0 - direction
	groundstation1 - direction
	star4 - direction
	groundstation3 - direction
	phenomenon5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 thermograph1)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star4)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon6)
	(supports instrument1 thermograph3)
	(calibration_target instrument1 star4)
	(supports instrument2 thermograph0)
	(supports instrument2 image2)
	(supports instrument2 thermograph3)
	(calibration_target instrument2 star0)
	(supports instrument3 image2)
	(calibration_target instrument3 groundstation3)
	(supports instrument4 image2)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 groundstation3)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument5 image2)
	(supports instrument5 thermograph3)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 groundstation3)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation2)
	(supports instrument6 thermograph3)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 thermograph3)
	(supports instrument7 thermograph1)
	(calibration_target instrument7 star4)
	(supports instrument8 image2)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 groundstation3)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon6)
)
(:goal (and
	(pointing satellite0 groundstation2)
	(have_image phenomenon5 thermograph0)
	(have_image phenomenon6 image2)
))

)
