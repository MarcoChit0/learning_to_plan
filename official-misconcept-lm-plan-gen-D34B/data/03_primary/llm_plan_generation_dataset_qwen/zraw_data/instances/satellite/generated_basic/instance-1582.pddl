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
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	satellite3 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite4 - satellite
	instrument11 - instrument
	satellite5 - satellite
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	infrared0 - mode
	star1 - direction
	groundstation0 - direction
	groundstation4 - direction
	star2 - direction
	groundstation3 - direction
	phenomenon5 - direction
	star6 - direction
	phenomenon7 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation3)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation4)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star2)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation3)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star1)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation3)
	(supports instrument5 infrared0)
	(calibration_target instrument5 groundstation3)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
	(supports instrument6 infrared0)
	(calibration_target instrument6 groundstation3)
	(supports instrument7 infrared0)
	(calibration_target instrument7 groundstation4)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon7)
	(supports instrument8 infrared0)
	(calibration_target instrument8 groundstation0)
	(supports instrument9 infrared0)
	(calibration_target instrument9 star2)
	(supports instrument10 infrared0)
	(calibration_target instrument10 star2)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation3)
	(supports instrument11 infrared0)
	(calibration_target instrument11 groundstation4)
	(on_board instrument11 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation4)
	(supports instrument12 infrared0)
	(calibration_target instrument12 groundstation3)
	(supports instrument13 infrared0)
	(calibration_target instrument13 star2)
	(supports instrument14 infrared0)
	(calibration_target instrument14 star2)
	(supports instrument15 infrared0)
	(calibration_target instrument15 groundstation3)
	(on_board instrument12 satellite5)
	(on_board instrument13 satellite5)
	(on_board instrument14 satellite5)
	(on_board instrument15 satellite5)
	(power_avail satellite5)
	(pointing satellite5 phenomenon7)
)
(:goal (and
	(pointing satellite2 phenomenon7)
	(pointing satellite5 star1)
	(have_image phenomenon5 infrared0)
	(have_image star6 infrared0)
	(have_image phenomenon7 infrared0)
))

)
