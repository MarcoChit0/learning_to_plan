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
	satellite2 - satellite
	instrument7 - instrument
	satellite3 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	infrared0 - mode
	groundstation3 - direction
	star2 - direction
	star0 - direction
	star5 - direction
	star4 - direction
	star1 - direction
	star6 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 star2)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star2)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star4)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation3)
	(supports instrument4 infrared0)
	(calibration_target instrument4 star5)
	(supports instrument5 infrared0)
	(calibration_target instrument5 star2)
	(calibration_target instrument5 star4)
	(supports instrument6 infrared0)
	(calibration_target instrument6 star0)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star5)
	(supports instrument7 infrared0)
	(calibration_target instrument7 star2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star5)
	(supports instrument8 infrared0)
	(calibration_target instrument8 star4)
	(supports instrument9 infrared0)
	(calibration_target instrument9 star0)
	(calibration_target instrument9 star4)
	(supports instrument10 infrared0)
	(calibration_target instrument10 star4)
	(calibration_target instrument10 star5)
	(supports instrument11 infrared0)
	(calibration_target instrument11 star1)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation3)
)
(:goal (and
	(pointing satellite3 star5)
	(have_image star6 infrared0)
))

)
