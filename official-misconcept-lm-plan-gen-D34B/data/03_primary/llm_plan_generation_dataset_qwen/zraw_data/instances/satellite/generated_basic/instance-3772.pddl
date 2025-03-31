(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	satellite2 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	satellite3 - satellite
	instrument8 - instrument
	satellite4 - satellite
	instrument9 - instrument
	instrument10 - instrument
	satellite5 - satellite
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	spectrograph2 - mode
	image1 - mode
	image0 - mode
	star0 - direction
	groundstation1 - direction
	star2 - direction
	planet3 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 image1)
	(calibration_target instrument0 star2)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument1 image0)
	(supports instrument1 image1)
	(supports instrument1 spectrograph2)
	(calibration_target instrument1 groundstation1)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
	(supports instrument2 image1)
	(supports instrument2 image0)
	(calibration_target instrument2 star0)
	(supports instrument3 spectrograph2)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 spectrograph2)
	(supports instrument4 image1)
	(supports instrument4 image0)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 image1)
	(supports instrument5 spectrograph2)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 image1)
	(supports instrument6 image0)
	(calibration_target instrument6 star2)
	(supports instrument7 image1)
	(calibration_target instrument7 groundstation1)
	(on_board instrument2 satellite2)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
	(supports instrument8 image0)
	(calibration_target instrument8 star2)
	(on_board instrument8 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation1)
	(supports instrument9 spectrograph2)
	(supports instrument9 image0)
	(supports instrument9 image1)
	(calibration_target instrument9 star0)
	(supports instrument10 image1)
	(supports instrument10 image0)
	(supports instrument10 spectrograph2)
	(calibration_target instrument10 groundstation1)
	(on_board instrument9 satellite4)
	(on_board instrument10 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star0)
	(supports instrument11 image1)
	(supports instrument11 spectrograph2)
	(supports instrument11 image0)
	(calibration_target instrument11 star0)
	(supports instrument12 image0)
	(supports instrument12 spectrograph2)
	(calibration_target instrument12 star2)
	(supports instrument13 image0)
	(calibration_target instrument13 star2)
	(supports instrument14 spectrograph2)
	(calibration_target instrument14 star0)
	(supports instrument15 image1)
	(supports instrument15 spectrograph2)
	(supports instrument15 image0)
	(calibration_target instrument15 groundstation1)
	(supports instrument16 image0)
	(calibration_target instrument16 star2)
	(on_board instrument11 satellite5)
	(on_board instrument12 satellite5)
	(on_board instrument13 satellite5)
	(on_board instrument14 satellite5)
	(on_board instrument15 satellite5)
	(on_board instrument16 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star0)
)
(:goal (and
	(pointing satellite0 groundstation1)
	(pointing satellite3 star2)
	(pointing satellite4 groundstation1)
	(have_image planet3 image1)
))

)
