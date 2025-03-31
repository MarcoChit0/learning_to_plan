(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite1 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
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
	satellite5 - satellite
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	image1 - mode
	spectrograph0 - mode
	infrared2 - mode
	star0 - direction
	star3 - direction
	groundstation2 - direction
	star1 - direction
	star4 - direction
	planet5 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(supports instrument0 infrared2)
	(supports instrument0 image1)
	(calibration_target instrument0 star1)
	(supports instrument1 image1)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 star0)
	(supports instrument2 infrared2)
	(calibration_target instrument2 star0)
	(supports instrument3 spectrograph0)
	(supports instrument3 infrared2)
	(supports instrument3 image1)
	(calibration_target instrument3 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument4 infrared2)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 star0)
	(supports instrument5 image1)
	(calibration_target instrument5 groundstation2)
	(supports instrument6 infrared2)
	(supports instrument6 spectrograph0)
	(supports instrument6 image1)
	(calibration_target instrument6 star1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 star3)
	(supports instrument8 infrared2)
	(supports instrument8 image1)
	(calibration_target instrument8 star0)
	(supports instrument9 spectrograph0)
	(supports instrument9 image1)
	(supports instrument9 infrared2)
	(calibration_target instrument9 star1)
	(supports instrument10 spectrograph0)
	(supports instrument10 infrared2)
	(supports instrument10 image1)
	(calibration_target instrument10 groundstation2)
	(supports instrument11 spectrograph0)
	(supports instrument11 image1)
	(supports instrument11 infrared2)
	(calibration_target instrument11 groundstation2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star1)
	(supports instrument12 spectrograph0)
	(calibration_target instrument12 groundstation2)
	(supports instrument13 spectrograph0)
	(supports instrument13 infrared2)
	(supports instrument13 image1)
	(calibration_target instrument13 groundstation2)
	(supports instrument14 spectrograph0)
	(supports instrument14 infrared2)
	(calibration_target instrument14 star1)
	(supports instrument15 spectrograph0)
	(supports instrument15 infrared2)
	(calibration_target instrument15 groundstation2)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star4)
	(supports instrument16 spectrograph0)
	(calibration_target instrument16 groundstation2)
	(on_board instrument16 satellite4)
	(power_avail satellite4)
	(pointing satellite4 planet5)
	(supports instrument17 infrared2)
	(supports instrument17 image1)
	(calibration_target instrument17 star3)
	(supports instrument18 infrared2)
	(supports instrument18 image1)
	(supports instrument18 spectrograph0)
	(calibration_target instrument18 groundstation2)
	(supports instrument19 image1)
	(supports instrument19 spectrograph0)
	(calibration_target instrument19 star1)
	(on_board instrument17 satellite5)
	(on_board instrument18 satellite5)
	(on_board instrument19 satellite5)
	(power_avail satellite5)
	(pointing satellite5 groundstation2)
)
(:goal (and
	(pointing satellite0 groundstation2)
	(pointing satellite5 star4)
	(have_image star4 spectrograph0)
	(have_image planet5 infrared2)
))

)
