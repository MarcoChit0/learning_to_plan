(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	satellite2 - satellite
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	satellite3 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	image1 - mode
	infrared0 - mode
	star0 - direction
	star1 - direction
	groundstation2 - direction
	star3 - direction
	planet4 - direction
)
(:init
	(supports instrument0 image1)
	(calibration_target instrument0 star3)
	(supports instrument1 image1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument2 image1)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star0)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
	(supports instrument3 image1)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 image1)
	(supports instrument4 infrared0)
	(calibration_target instrument4 star3)
	(supports instrument5 image1)
	(calibration_target instrument5 star3)
	(supports instrument6 image1)
	(calibration_target instrument6 star0)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet4)
	(supports instrument7 image1)
	(calibration_target instrument7 star1)
	(supports instrument8 infrared0)
	(supports instrument8 image1)
	(calibration_target instrument8 groundstation2)
	(supports instrument9 image1)
	(supports instrument9 infrared0)
	(calibration_target instrument9 star3)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star1)
)
(:goal (and
	(pointing satellite0 planet4)
	(have_image planet4 infrared0)
))

)
