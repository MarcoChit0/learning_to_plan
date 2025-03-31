(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	satellite2 - satellite
	instrument8 - instrument
	satellite3 - satellite
	instrument9 - instrument
	image1 - mode
	infrared0 - mode
	star1 - direction
	groundstation0 - direction
	star2 - direction
	star3 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 star1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument3 infrared0)
	(supports instrument3 image1)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 image1)
	(calibration_target instrument4 star1)
	(supports instrument5 image1)
	(supports instrument5 infrared0)
	(calibration_target instrument5 star1)
	(supports instrument6 image1)
	(supports instrument6 infrared0)
	(calibration_target instrument6 star1)
	(supports instrument7 infrared0)
	(supports instrument7 image1)
	(calibration_target instrument7 groundstation0)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument8 infrared0)
	(supports instrument8 image1)
	(calibration_target instrument8 star1)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
	(supports instrument9 infrared0)
	(supports instrument9 image1)
	(calibration_target instrument9 groundstation0)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star2)
)
(:goal (and
	(pointing satellite0 star1)
	(have_image star2 image1)
	(have_image star3 image1)
))

)
