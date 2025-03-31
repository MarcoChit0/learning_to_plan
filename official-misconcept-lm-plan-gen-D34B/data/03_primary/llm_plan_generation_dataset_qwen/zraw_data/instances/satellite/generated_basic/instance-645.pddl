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
	instrument8 - instrument
	satellite2 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	satellite3 - satellite
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	satellite4 - satellite
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	instrument20 - instrument
	spectrograph0 - mode
	image2 - mode
	image1 - mode
	star1 - direction
	star2 - direction
	groundstation3 - direction
	star0 - direction
	planet4 - direction
	phenomenon5 - direction
	star6 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 image2)
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 star1)
	(supports instrument1 image2)
	(calibration_target instrument1 groundstation3)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star6)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 star1)
	(supports instrument4 image1)
	(supports instrument4 image2)
	(calibration_target instrument4 star1)
	(supports instrument5 spectrograph0)
	(supports instrument5 image2)
	(supports instrument5 image1)
	(calibration_target instrument5 star1)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 star0)
	(supports instrument7 image2)
	(supports instrument7 spectrograph0)
	(supports instrument7 image1)
	(calibration_target instrument7 star2)
	(supports instrument8 spectrograph0)
	(supports instrument8 image1)
	(supports instrument8 image2)
	(calibration_target instrument8 star0)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument9 image2)
	(supports instrument9 image1)
	(calibration_target instrument9 star0)
	(supports instrument10 image2)
	(calibration_target instrument10 star0)
	(supports instrument11 image1)
	(supports instrument11 spectrograph0)
	(supports instrument11 image2)
	(calibration_target instrument11 star2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
	(supports instrument12 image2)
	(supports instrument12 image1)
	(calibration_target instrument12 groundstation3)
	(supports instrument13 spectrograph0)
	(calibration_target instrument13 star0)
	(supports instrument14 spectrograph0)
	(supports instrument14 image2)
	(calibration_target instrument14 groundstation3)
	(supports instrument15 image2)
	(calibration_target instrument15 star1)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation3)
	(supports instrument16 image2)
	(supports instrument16 image1)
	(supports instrument16 spectrograph0)
	(calibration_target instrument16 star1)
	(supports instrument17 spectrograph0)
	(supports instrument17 image1)
	(calibration_target instrument17 groundstation3)
	(supports instrument18 image1)
	(supports instrument18 image2)
	(calibration_target instrument18 star2)
	(supports instrument19 image2)
	(supports instrument19 spectrograph0)
	(calibration_target instrument19 groundstation3)
	(supports instrument20 image2)
	(supports instrument20 image1)
	(calibration_target instrument20 star0)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(on_board instrument19 satellite4)
	(on_board instrument20 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star1)
)
(:goal (and
	(pointing satellite1 groundstation3)
	(pointing satellite2 groundstation3)
	(pointing satellite4 star2)
	(have_image planet4 image2)
	(have_image phenomenon5 spectrograph0)
	(have_image star6 image2)
))

)
