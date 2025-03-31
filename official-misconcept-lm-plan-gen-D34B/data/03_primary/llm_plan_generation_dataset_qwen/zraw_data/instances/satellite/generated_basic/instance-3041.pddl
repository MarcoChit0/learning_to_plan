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
	satellite3 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	satellite4 - satellite
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	image1 - mode
	spectrograph0 - mode
	star0 - direction
	star1 - direction
	groundstation2 - direction
	star3 - direction
	planet4 - direction
	phenomenon5 - direction
	star6 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 star3)
	(supports instrument1 spectrograph0)
	(supports instrument1 image1)
	(calibration_target instrument1 star1)
	(supports instrument2 spectrograph0)
	(supports instrument2 image1)
	(calibration_target instrument2 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument3 spectrograph0)
	(supports instrument3 image1)
	(calibration_target instrument3 star3)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star6)
	(supports instrument4 spectrograph0)
	(supports instrument4 image1)
	(calibration_target instrument4 groundstation2)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 star0)
	(supports instrument6 spectrograph0)
	(supports instrument6 image1)
	(calibration_target instrument6 star3)
	(supports instrument7 image1)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 star0)
	(supports instrument8 image1)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 star1)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 star1)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
	(supports instrument10 image1)
	(supports instrument10 spectrograph0)
	(calibration_target instrument10 star0)
	(supports instrument11 spectrograph0)
	(supports instrument11 image1)
	(calibration_target instrument11 star1)
	(supports instrument12 spectrograph0)
	(supports instrument12 image1)
	(calibration_target instrument12 star3)
	(supports instrument13 spectrograph0)
	(calibration_target instrument13 groundstation2)
	(supports instrument14 spectrograph0)
	(supports instrument14 image1)
	(calibration_target instrument14 star3)
	(supports instrument15 spectrograph0)
	(supports instrument15 image1)
	(calibration_target instrument15 star1)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet4)
	(supports instrument16 image1)
	(supports instrument16 spectrograph0)
	(calibration_target instrument16 star0)
	(supports instrument17 image1)
	(supports instrument17 spectrograph0)
	(calibration_target instrument17 star1)
	(supports instrument18 spectrograph0)
	(supports instrument18 image1)
	(calibration_target instrument18 groundstation2)
	(supports instrument19 spectrograph0)
	(supports instrument19 image1)
	(calibration_target instrument19 star3)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(on_board instrument19 satellite4)
	(power_avail satellite4)
	(pointing satellite4 phenomenon5)
)
(:goal (and
	(have_image planet4 spectrograph0)
	(have_image phenomenon5 spectrograph0)
	(have_image star6 spectrograph0)
))

)
