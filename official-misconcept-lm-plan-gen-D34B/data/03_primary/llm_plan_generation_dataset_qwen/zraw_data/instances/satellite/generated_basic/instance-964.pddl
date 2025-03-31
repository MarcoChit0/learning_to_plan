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
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	infrared0 - mode
	infrared3 - mode
	image1 - mode
	thermograph2 - mode
	star2 - direction
	groundstation1 - direction
	star3 - direction
	star0 - direction
	phenomenon4 - direction
	planet5 - direction
	star6 - direction
)
(:init
	(supports instrument0 image1)
	(calibration_target instrument0 star3)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 star3)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 thermograph2)
	(supports instrument3 infrared3)
	(calibration_target instrument3 star2)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 thermograph2)
	(supports instrument5 image1)
	(supports instrument5 infrared0)
	(calibration_target instrument5 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument6 infrared3)
	(supports instrument6 infrared0)
	(supports instrument6 image1)
	(calibration_target instrument6 star3)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
	(supports instrument7 infrared0)
	(calibration_target instrument7 star3)
	(supports instrument8 infrared3)
	(supports instrument8 infrared0)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 infrared3)
	(supports instrument9 image1)
	(calibration_target instrument9 groundstation1)
	(supports instrument10 thermograph2)
	(supports instrument10 infrared3)
	(calibration_target instrument10 star3)
	(supports instrument11 thermograph2)
	(supports instrument11 infrared3)
	(supports instrument11 image1)
	(calibration_target instrument11 star3)
	(supports instrument12 infrared0)
	(supports instrument12 image1)
	(calibration_target instrument12 star0)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
)
(:goal (and
	(pointing satellite1 star2)
	(pointing satellite2 planet5)
	(have_image phenomenon4 thermograph2)
	(have_image planet5 image1)
	(have_image star6 thermograph2)
))

)
