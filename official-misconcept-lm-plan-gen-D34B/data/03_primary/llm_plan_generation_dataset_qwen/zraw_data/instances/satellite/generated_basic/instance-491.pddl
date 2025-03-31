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
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite2 - satellite
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	satellite3 - satellite
	instrument14 - instrument
	thermograph3 - mode
	thermograph0 - mode
	thermograph1 - mode
	image2 - mode
	star4 - direction
	groundstation2 - direction
	star1 - direction
	groundstation3 - direction
	star0 - direction
	phenomenon5 - direction
	phenomenon6 - direction
	phenomenon7 - direction
)
(:init
	(supports instrument0 image2)
	(supports instrument0 thermograph3)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star1)
	(supports instrument1 thermograph3)
	(supports instrument1 thermograph1)
	(supports instrument1 image2)
	(calibration_target instrument1 star1)
	(supports instrument2 thermograph1)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star0)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 groundstation3)
	(supports instrument4 thermograph3)
	(calibration_target instrument4 star1)
	(supports instrument5 image2)
	(calibration_target instrument5 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation3)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 star0)
	(supports instrument7 thermograph0)
	(supports instrument7 thermograph1)
	(calibration_target instrument7 groundstation2)
	(supports instrument8 image2)
	(calibration_target instrument8 groundstation3)
	(supports instrument9 thermograph0)
	(supports instrument9 thermograph3)
	(supports instrument9 thermograph1)
	(calibration_target instrument9 star0)
	(supports instrument10 thermograph3)
	(supports instrument10 image2)
	(supports instrument10 thermograph0)
	(calibration_target instrument10 groundstation3)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(on_board instrument10 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument11 image2)
	(supports instrument11 thermograph1)
	(supports instrument11 thermograph3)
	(calibration_target instrument11 star1)
	(supports instrument12 thermograph1)
	(supports instrument12 image2)
	(supports instrument12 thermograph3)
	(calibration_target instrument12 star1)
	(supports instrument13 thermograph3)
	(calibration_target instrument13 groundstation3)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation3)
	(supports instrument14 image2)
	(supports instrument14 thermograph3)
	(supports instrument14 thermograph0)
	(calibration_target instrument14 star0)
	(on_board instrument14 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star1)
)
(:goal (and
	(pointing satellite0 phenomenon6)
	(pointing satellite1 groundstation2)
	(pointing satellite3 phenomenon5)
	(have_image phenomenon5 thermograph1)
	(have_image phenomenon6 thermograph0)
	(have_image phenomenon7 thermograph0)
))

)
