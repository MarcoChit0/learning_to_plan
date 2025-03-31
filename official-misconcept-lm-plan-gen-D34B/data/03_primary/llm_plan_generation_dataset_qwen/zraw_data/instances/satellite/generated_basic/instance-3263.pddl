(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite1 - satellite
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	satellite3 - satellite
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	image1 - mode
	image0 - mode
	thermograph2 - mode
	groundstation2 - direction
	groundstation0 - direction
	groundstation1 - direction
	phenomenon3 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 image1)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 image0)
	(supports instrument1 image1)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 image0)
	(supports instrument2 image1)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 thermograph2)
	(supports instrument3 image1)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 image0)
	(supports instrument4 image1)
	(supports instrument4 thermograph2)
	(calibration_target instrument4 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument5 image0)
	(supports instrument5 thermograph2)
	(supports instrument5 image1)
	(calibration_target instrument5 groundstation0)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument6 image1)
	(supports instrument6 image0)
	(supports instrument6 thermograph2)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 image0)
	(supports instrument7 image1)
	(supports instrument7 thermograph2)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 image0)
	(supports instrument8 thermograph2)
	(supports instrument8 image1)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 thermograph2)
	(supports instrument9 image0)
	(calibration_target instrument9 groundstation2)
	(supports instrument10 image0)
	(calibration_target instrument10 groundstation0)
	(supports instrument11 image0)
	(calibration_target instrument11 groundstation2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation2)
	(supports instrument12 thermograph2)
	(supports instrument12 image0)
	(supports instrument12 image1)
	(calibration_target instrument12 groundstation0)
	(supports instrument13 thermograph2)
	(supports instrument13 image0)
	(calibration_target instrument13 groundstation0)
	(supports instrument14 thermograph2)
	(calibration_target instrument14 groundstation0)
	(supports instrument15 image1)
	(calibration_target instrument15 groundstation1)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation1)
)
(:goal (and
	(have_image phenomenon3 image0)
))

)
