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
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	satellite2 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	infrared0 - mode
	infrared1 - mode
	thermograph2 - mode
	groundstation4 - direction
	groundstation1 - direction
	groundstation3 - direction
	groundstation0 - direction
	star2 - direction
	groundstation5 - direction
	star6 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation4)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 star2)
	(supports instrument2 infrared1)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation1)
	(calibration_target instrument2 groundstation4)
	(supports instrument3 infrared0)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation4)
	(supports instrument4 infrared1)
	(calibration_target instrument4 groundstation5)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 thermograph2)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 infrared0)
	(supports instrument6 thermograph2)
	(supports instrument6 infrared1)
	(calibration_target instrument6 star2)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 infrared1)
	(supports instrument7 infrared0)
	(supports instrument7 thermograph2)
	(calibration_target instrument7 groundstation4)
	(calibration_target instrument7 groundstation1)
	(supports instrument8 infrared0)
	(supports instrument8 infrared1)
	(calibration_target instrument8 star2)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
	(supports instrument9 thermograph2)
	(supports instrument9 infrared1)
	(supports instrument9 infrared0)
	(calibration_target instrument9 groundstation1)
	(calibration_target instrument9 groundstation0)
	(supports instrument10 thermograph2)
	(supports instrument10 infrared1)
	(supports instrument10 infrared0)
	(calibration_target instrument10 groundstation3)
	(calibration_target instrument10 groundstation0)
	(supports instrument11 thermograph2)
	(calibration_target instrument11 groundstation0)
	(supports instrument12 infrared0)
	(supports instrument12 infrared1)
	(calibration_target instrument12 groundstation5)
	(calibration_target instrument12 star2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
)
(:goal (and
	(pointing satellite2 star2)
	(have_image star6 infrared1)
))

)
