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
	satellite2 - satellite
	instrument6 - instrument
	satellite3 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite4 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	image0 - mode
	thermograph2 - mode
	image3 - mode
	image1 - mode
	star1 - direction
	groundstation3 - direction
	groundstation2 - direction
	star0 - direction
	phenomenon4 - direction
	planet5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 star1)
	(supports instrument1 thermograph2)
	(supports instrument1 image0)
	(supports instrument1 image3)
	(calibration_target instrument1 star1)
	(supports instrument2 image3)
	(supports instrument2 thermograph2)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 image3)
	(supports instrument3 image1)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument4 image3)
	(supports instrument4 image1)
	(calibration_target instrument4 star1)
	(supports instrument5 image3)
	(supports instrument5 thermograph2)
	(supports instrument5 image1)
	(calibration_target instrument5 groundstation3)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument6 image0)
	(supports instrument6 thermograph2)
	(calibration_target instrument6 groundstation2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon6)
	(supports instrument7 image3)
	(calibration_target instrument7 groundstation2)
	(supports instrument8 image1)
	(supports instrument8 thermograph2)
	(supports instrument8 image3)
	(calibration_target instrument8 groundstation2)
	(supports instrument9 image1)
	(supports instrument9 image3)
	(supports instrument9 thermograph2)
	(calibration_target instrument9 groundstation3)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star1)
	(supports instrument10 thermograph2)
	(supports instrument10 image0)
	(supports instrument10 image3)
	(calibration_target instrument10 groundstation2)
	(supports instrument11 image3)
	(calibration_target instrument11 star0)
	(supports instrument12 image3)
	(calibration_target instrument12 star0)
	(supports instrument13 image3)
	(supports instrument13 thermograph2)
	(calibration_target instrument13 star0)
	(on_board instrument10 satellite4)
	(on_board instrument11 satellite4)
	(on_board instrument12 satellite4)
	(on_board instrument13 satellite4)
	(power_avail satellite4)
	(pointing satellite4 phenomenon4)
)
(:goal (and
	(pointing satellite4 star1)
	(have_image phenomenon4 image3)
	(have_image planet5 thermograph2)
	(have_image phenomenon6 thermograph2)
))

)
