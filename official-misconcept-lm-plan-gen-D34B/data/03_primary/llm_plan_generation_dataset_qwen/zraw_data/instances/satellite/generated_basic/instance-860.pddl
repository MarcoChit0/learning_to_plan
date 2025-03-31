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
	satellite2 - satellite
	instrument8 - instrument
	satellite3 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	satellite4 - satellite
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	satellite5 - satellite
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	infrared0 - mode
	image1 - mode
	spectrograph2 - mode
	star3 - direction
	groundstation0 - direction
	star2 - direction
	groundstation1 - direction
	planet4 - direction
	planet5 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 infrared0)
	(calibration_target instrument0 star3)
	(supports instrument1 image1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star2)
	(supports instrument2 image1)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star2)
	(supports instrument3 image1)
	(supports instrument3 spectrograph2)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 image1)
	(calibration_target instrument4 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument5 spectrograph2)
	(supports instrument5 infrared0)
	(supports instrument5 image1)
	(calibration_target instrument5 star2)
	(supports instrument6 infrared0)
	(supports instrument6 image1)
	(supports instrument6 spectrograph2)
	(calibration_target instrument6 star3)
	(supports instrument7 image1)
	(supports instrument7 infrared0)
	(calibration_target instrument7 groundstation0)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet5)
	(supports instrument8 infrared0)
	(calibration_target instrument8 star3)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
	(supports instrument9 spectrograph2)
	(calibration_target instrument9 star2)
	(supports instrument10 spectrograph2)
	(calibration_target instrument10 groundstation1)
	(supports instrument11 image1)
	(supports instrument11 infrared0)
	(calibration_target instrument11 groundstation1)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet5)
	(supports instrument12 infrared0)
	(supports instrument12 spectrograph2)
	(supports instrument12 image1)
	(calibration_target instrument12 groundstation1)
	(supports instrument13 infrared0)
	(supports instrument13 spectrograph2)
	(calibration_target instrument13 star3)
	(supports instrument14 image1)
	(supports instrument14 infrared0)
	(supports instrument14 spectrograph2)
	(calibration_target instrument14 groundstation0)
	(supports instrument15 infrared0)
	(supports instrument15 image1)
	(supports instrument15 spectrograph2)
	(calibration_target instrument15 star2)
	(supports instrument16 spectrograph2)
	(calibration_target instrument16 groundstation1)
	(on_board instrument12 satellite4)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(power_avail satellite4)
	(pointing satellite4 planet4)
	(supports instrument17 infrared0)
	(supports instrument17 spectrograph2)
	(calibration_target instrument17 groundstation1)
	(supports instrument18 infrared0)
	(calibration_target instrument18 groundstation1)
	(supports instrument19 spectrograph2)
	(calibration_target instrument19 groundstation1)
	(on_board instrument17 satellite5)
	(on_board instrument18 satellite5)
	(on_board instrument19 satellite5)
	(power_avail satellite5)
	(pointing satellite5 planet4)
)
(:goal (and
	(pointing satellite1 planet4)
	(pointing satellite4 groundstation0)
	(have_image planet4 image1)
	(have_image planet5 infrared0)
))

)
