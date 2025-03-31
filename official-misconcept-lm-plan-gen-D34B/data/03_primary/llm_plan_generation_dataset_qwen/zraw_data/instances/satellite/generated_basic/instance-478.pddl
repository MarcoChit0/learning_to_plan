(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite1 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite2 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	satellite3 - satellite
	instrument16 - instrument
	image1 - mode
	spectrograph2 - mode
	image0 - mode
	infrared3 - mode
	star3 - direction
	star4 - direction
	groundstation2 - direction
	groundstation1 - direction
	groundstation0 - direction
	planet5 - direction
	planet6 - direction
	planet7 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 star4)
	(supports instrument1 image1)
	(supports instrument1 spectrograph2)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 spectrograph2)
	(supports instrument2 image1)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 infrared3)
	(supports instrument4 image0)
	(supports instrument4 image1)
	(calibration_target instrument4 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument5 image1)
	(supports instrument5 infrared3)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 infrared3)
	(calibration_target instrument6 star3)
	(supports instrument7 image1)
	(calibration_target instrument7 star3)
	(supports instrument8 spectrograph2)
	(calibration_target instrument8 groundstation2)
	(supports instrument9 image0)
	(calibration_target instrument9 groundstation1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument10 infrared3)
	(supports instrument10 spectrograph2)
	(supports instrument10 image1)
	(calibration_target instrument10 star4)
	(supports instrument11 image1)
	(supports instrument11 spectrograph2)
	(calibration_target instrument11 star4)
	(supports instrument12 image0)
	(supports instrument12 spectrograph2)
	(supports instrument12 infrared3)
	(calibration_target instrument12 groundstation0)
	(supports instrument13 image0)
	(supports instrument13 infrared3)
	(calibration_target instrument13 groundstation2)
	(supports instrument14 spectrograph2)
	(calibration_target instrument14 groundstation0)
	(supports instrument15 spectrograph2)
	(calibration_target instrument15 groundstation1)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(on_board instrument14 satellite2)
	(on_board instrument15 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
	(supports instrument16 infrared3)
	(supports instrument16 image1)
	(calibration_target instrument16 groundstation0)
	(on_board instrument16 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet6)
)
(:goal (and
	(pointing satellite1 star4)
	(pointing satellite2 star4)
	(pointing satellite3 star4)
	(have_image planet5 image0)
	(have_image planet6 image0)
	(have_image planet7 image1)
))

)
