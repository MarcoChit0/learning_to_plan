(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	satellite3 - satellite
	instrument8 - instrument
	satellite4 - satellite
	instrument9 - instrument
	satellite5 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	infrared0 - mode
	groundstation2 - direction
	star1 - direction
	groundstation3 - direction
	star0 - direction
	star4 - direction
	star5 - direction
	star6 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 star1)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star5)
	(calibration_target instrument1 star0)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star0)
	(calibration_target instrument2 star4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star4)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument4 infrared0)
	(calibration_target instrument4 star4)
	(calibration_target instrument4 star5)
	(supports instrument5 infrared0)
	(calibration_target instrument5 groundstation3)
	(supports instrument6 infrared0)
	(calibration_target instrument6 star0)
	(supports instrument7 infrared0)
	(calibration_target instrument7 star0)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star4)
	(supports instrument8 infrared0)
	(calibration_target instrument8 star0)
	(on_board instrument8 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star1)
	(supports instrument9 infrared0)
	(calibration_target instrument9 groundstation3)
	(calibration_target instrument9 star0)
	(on_board instrument9 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation3)
	(supports instrument10 infrared0)
	(calibration_target instrument10 star0)
	(supports instrument11 infrared0)
	(calibration_target instrument11 star4)
	(calibration_target instrument11 star5)
	(supports instrument12 infrared0)
	(calibration_target instrument12 star5)
	(on_board instrument10 satellite5)
	(on_board instrument11 satellite5)
	(on_board instrument12 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star5)
)
(:goal (and
	(pointing satellite2 star4)
	(pointing satellite3 star0)
	(pointing satellite4 star0)
	(pointing satellite5 groundstation2)
	(have_image star6 infrared0)
))

)
