(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	satellite3 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite4 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	satellite5 - satellite
	instrument16 - instrument
	image1 - mode
	image2 - mode
	spectrograph0 - mode
	star1 - direction
	groundstation3 - direction
	star0 - direction
	star2 - direction
	star4 - direction
	star5 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(supports instrument0 image1)
	(supports instrument0 image2)
	(calibration_target instrument0 groundstation3)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument1 image2)
	(supports instrument1 spectrograph0)
	(supports instrument1 image1)
	(calibration_target instrument1 star2)
	(supports instrument2 image1)
	(supports instrument2 image2)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 star0)
	(supports instrument3 image2)
	(supports instrument3 image1)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 star0)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument4 image2)
	(supports instrument4 image1)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 star2)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 star1)
	(supports instrument6 image2)
	(supports instrument6 image1)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 star1)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
	(supports instrument7 image1)
	(calibration_target instrument7 star2)
	(supports instrument8 image2)
	(calibration_target instrument8 groundstation3)
	(supports instrument9 spectrograph0)
	(supports instrument9 image2)
	(calibration_target instrument9 star0)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star5)
	(supports instrument10 image2)
	(supports instrument10 spectrograph0)
	(calibration_target instrument10 star1)
	(supports instrument11 spectrograph0)
	(supports instrument11 image2)
	(calibration_target instrument11 star1)
	(supports instrument12 image1)
	(calibration_target instrument12 star2)
	(supports instrument13 spectrograph0)
	(supports instrument13 image2)
	(calibration_target instrument13 groundstation3)
	(supports instrument14 image1)
	(supports instrument14 image2)
	(calibration_target instrument14 star0)
	(supports instrument15 image1)
	(supports instrument15 spectrograph0)
	(supports instrument15 image2)
	(calibration_target instrument15 star0)
	(on_board instrument10 satellite4)
	(on_board instrument11 satellite4)
	(on_board instrument12 satellite4)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star5)
	(supports instrument16 image2)
	(supports instrument16 spectrograph0)
	(calibration_target instrument16 star2)
	(on_board instrument16 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star2)
)
(:goal (and
	(pointing satellite0 star0)
	(pointing satellite3 star2)
	(pointing satellite4 star0)
	(have_image star4 image2)
	(have_image star5 image2)
))

)
