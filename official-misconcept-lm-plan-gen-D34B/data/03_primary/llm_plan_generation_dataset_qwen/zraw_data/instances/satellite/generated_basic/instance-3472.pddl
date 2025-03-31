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
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	infrared0 - mode
	image1 - mode
	star2 - direction
	groundstation1 - direction
	star0 - direction
	planet3 - direction
	star4 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 star0)
	(supports instrument1 image1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet3)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star2)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star0)
	(supports instrument4 image1)
	(supports instrument4 infrared0)
	(calibration_target instrument4 star0)
	(supports instrument5 image1)
	(supports instrument5 infrared0)
	(calibration_target instrument5 star2)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
	(supports instrument6 image1)
	(supports instrument6 infrared0)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 infrared0)
	(supports instrument7 image1)
	(calibration_target instrument7 groundstation1)
	(supports instrument8 infrared0)
	(supports instrument8 image1)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 infrared0)
	(supports instrument9 image1)
	(calibration_target instrument9 star0)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
)
(:goal (and
	(have_image planet3 infrared0)
	(have_image star4 infrared0)
))

)
