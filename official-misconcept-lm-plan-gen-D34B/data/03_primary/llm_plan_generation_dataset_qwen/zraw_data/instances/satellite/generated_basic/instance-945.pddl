(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	infrared0 - mode
	thermograph2 - mode
	infrared1 - mode
	image3 - mode
	star0 - direction
	groundstation2 - direction
	groundstation3 - direction
	groundstation1 - direction
	phenomenon4 - direction
	phenomenon5 - direction
	star6 - direction
)
(:init
	(supports instrument0 infrared0)
	(supports instrument0 thermograph2)
	(calibration_target instrument0 star0)
	(supports instrument1 infrared0)
	(supports instrument1 thermograph2)
	(supports instrument1 image3)
	(calibration_target instrument1 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon4)
	(supports instrument2 image3)
	(supports instrument2 thermograph2)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star0)
	(supports instrument3 infrared0)
	(supports instrument3 image3)
	(supports instrument3 infrared1)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 thermograph2)
	(supports instrument4 image3)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation2)
	(supports instrument5 image3)
	(supports instrument5 infrared0)
	(supports instrument5 thermograph2)
	(calibration_target instrument5 groundstation2)
	(supports instrument6 thermograph2)
	(supports instrument6 infrared1)
	(supports instrument6 image3)
	(calibration_target instrument6 star0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon4)
	(supports instrument7 infrared0)
	(supports instrument7 infrared1)
	(calibration_target instrument7 groundstation3)
	(supports instrument8 infrared0)
	(calibration_target instrument8 groundstation2)
	(supports instrument9 infrared0)
	(supports instrument9 image3)
	(calibration_target instrument9 groundstation2)
	(supports instrument10 infrared1)
	(calibration_target instrument10 groundstation3)
	(supports instrument11 infrared0)
	(calibration_target instrument11 groundstation1)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon4)
)
(:goal (and
	(pointing satellite1 groundstation3)
	(have_image phenomenon4 thermograph2)
	(have_image phenomenon5 thermograph2)
	(have_image star6 thermograph2)
))

)
