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
	instrument6 - instrument
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	infrared0 - mode
	image1 - mode
	infrared3 - mode
	image2 - mode
	groundstation0 - direction
	groundstation2 - direction
	star3 - direction
	groundstation1 - direction
	star4 - direction
	planet5 - direction
	planet6 - direction
)
(:init
	(supports instrument0 infrared0)
	(supports instrument0 image1)
	(supports instrument0 image2)
	(calibration_target instrument0 groundstation2)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument1 image1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star3)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 infrared0)
	(supports instrument3 image2)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 infrared3)
	(supports instrument4 image1)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 image2)
	(supports instrument5 infrared0)
	(calibration_target instrument5 star3)
	(supports instrument6 image2)
	(supports instrument6 image1)
	(calibration_target instrument6 groundstation1)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument7 infrared0)
	(supports instrument7 image2)
	(supports instrument7 image1)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 image2)
	(supports instrument8 infrared0)
	(supports instrument8 image1)
	(calibration_target instrument8 groundstation2)
	(supports instrument9 infrared0)
	(supports instrument9 infrared3)
	(calibration_target instrument9 star3)
	(supports instrument10 image2)
	(supports instrument10 infrared3)
	(calibration_target instrument10 groundstation1)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet5)
)
(:goal (and
	(pointing satellite0 groundstation1)
	(pointing satellite2 planet5)
	(have_image star4 image2)
	(have_image planet5 infrared3)
	(have_image planet6 image1)
))

)
