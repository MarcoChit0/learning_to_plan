(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite3 - satellite
	instrument11 - instrument
	instrument12 - instrument
	satellite4 - satellite
	instrument13 - instrument
	instrument14 - instrument
	thermograph2 - mode
	thermograph0 - mode
	image3 - mode
	image1 - mode
	groundstation3 - direction
	star0 - direction
	groundstation1 - direction
	star2 - direction
	phenomenon4 - direction
	planet5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star2)
	(supports instrument1 image3)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 groundstation3)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument3 image1)
	(supports instrument3 image3)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 groundstation3)
	(supports instrument4 thermograph0)
	(supports instrument4 image3)
	(supports instrument4 thermograph2)
	(calibration_target instrument4 groundstation3)
	(supports instrument5 image1)
	(supports instrument5 image3)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 star2)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon4)
	(supports instrument6 image1)
	(calibration_target instrument6 star0)
	(supports instrument7 image1)
	(calibration_target instrument7 star0)
	(supports instrument8 image1)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 star2)
	(supports instrument9 thermograph0)
	(supports instrument9 image3)
	(supports instrument9 thermograph2)
	(calibration_target instrument9 star0)
	(supports instrument10 image1)
	(calibration_target instrument10 star2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
	(supports instrument11 thermograph2)
	(supports instrument11 image3)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 groundstation3)
	(supports instrument12 thermograph2)
	(supports instrument12 image1)
	(supports instrument12 thermograph0)
	(calibration_target instrument12 star0)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet5)
	(supports instrument13 thermograph2)
	(calibration_target instrument13 groundstation1)
	(supports instrument14 thermograph0)
	(calibration_target instrument14 star2)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(power_avail satellite4)
	(pointing satellite4 phenomenon6)
)
(:goal (and
	(have_image phenomenon4 image3)
	(have_image planet5 image3)
	(have_image phenomenon6 thermograph2)
))

)
