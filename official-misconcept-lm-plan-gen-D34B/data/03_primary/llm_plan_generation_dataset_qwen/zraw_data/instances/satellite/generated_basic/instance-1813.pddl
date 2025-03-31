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
	instrument8 - instrument
	instrument9 - instrument
	image2 - mode
	spectrograph0 - mode
	infrared3 - mode
	image1 - mode
	groundstation0 - direction
	star2 - direction
	groundstation1 - direction
	planet3 - direction
	star4 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 infrared3)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 infrared3)
	(calibration_target instrument2 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument3 image2)
	(calibration_target instrument3 groundstation0)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet3)
	(supports instrument4 image1)
	(supports instrument4 spectrograph0)
	(supports instrument4 infrared3)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 image1)
	(supports instrument5 infrared3)
	(supports instrument5 image2)
	(calibration_target instrument5 star2)
	(supports instrument6 image2)
	(supports instrument6 image1)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 image1)
	(calibration_target instrument7 groundstation1)
	(supports instrument8 infrared3)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 image1)
	(supports instrument9 spectrograph0)
	(supports instrument9 infrared3)
	(calibration_target instrument9 groundstation1)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
)
(:goal (and
	(have_image planet3 image1)
	(have_image star4 infrared3)
))

)
