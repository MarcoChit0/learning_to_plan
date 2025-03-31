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
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	satellite3 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	spectrograph1 - mode
	infrared3 - mode
	image0 - mode
	image2 - mode
	star0 - direction
	groundstation1 - direction
	star2 - direction
	star3 - direction
	star4 - direction
	star5 - direction
	planet6 - direction
	planet7 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 star0)
	(supports instrument1 infrared3)
	(supports instrument1 image2)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument2 infrared3)
	(calibration_target instrument2 star2)
	(supports instrument3 infrared3)
	(supports instrument3 image0)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 image0)
	(supports instrument4 image2)
	(calibration_target instrument4 star0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet6)
	(supports instrument5 infrared3)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 infrared3)
	(supports instrument6 spectrograph1)
	(calibration_target instrument6 groundstation1)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
	(supports instrument7 infrared3)
	(supports instrument7 image0)
	(supports instrument7 spectrograph1)
	(calibration_target instrument7 groundstation1)
	(supports instrument8 image2)
	(supports instrument8 infrared3)
	(calibration_target instrument8 star2)
	(supports instrument9 image0)
	(calibration_target instrument9 star2)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet6)
)
(:goal (and
	(pointing satellite3 planet7)
	(have_image star3 image2)
	(have_image star4 image2)
	(have_image star5 image2)
	(have_image planet6 infrared3)
	(have_image planet7 infrared3)
))

)
