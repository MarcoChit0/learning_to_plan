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
	satellite2 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	image1 - mode
	thermograph3 - mode
	image2 - mode
	infrared0 - mode
	groundstation2 - direction
	groundstation1 - direction
	groundstation0 - direction
	phenomenon3 - direction
	phenomenon4 - direction
)
(:init
	(supports instrument0 thermograph3)
	(supports instrument0 infrared0)
	(supports instrument0 image2)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 image1)
	(supports instrument1 thermograph3)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 infrared0)
	(supports instrument2 image2)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 infrared0)
	(supports instrument3 image1)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 image2)
	(supports instrument4 image1)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 image2)
	(supports instrument5 infrared0)
	(calibration_target instrument5 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon4)
	(supports instrument6 thermograph3)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 image2)
	(supports instrument7 infrared0)
	(supports instrument7 image1)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 infrared0)
	(calibration_target instrument8 groundstation1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon4)
	(supports instrument9 image2)
	(calibration_target instrument9 groundstation1)
	(supports instrument10 thermograph3)
	(supports instrument10 infrared0)
	(supports instrument10 image2)
	(calibration_target instrument10 groundstation0)
	(supports instrument11 image1)
	(calibration_target instrument11 groundstation2)
	(supports instrument12 thermograph3)
	(supports instrument12 image2)
	(calibration_target instrument12 groundstation0)
	(supports instrument13 thermograph3)
	(calibration_target instrument13 groundstation1)
	(supports instrument14 thermograph3)
	(supports instrument14 infrared0)
	(supports instrument14 image1)
	(calibration_target instrument14 groundstation0)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(on_board instrument14 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation2)
)
(:goal (and
	(pointing satellite0 groundstation2)
	(pointing satellite2 phenomenon4)
	(have_image phenomenon3 infrared0)
	(have_image phenomenon4 image1)
))

)
