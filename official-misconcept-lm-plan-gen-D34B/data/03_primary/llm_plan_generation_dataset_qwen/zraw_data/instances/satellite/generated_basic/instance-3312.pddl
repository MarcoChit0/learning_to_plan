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
	instrument11 - instrument
	instrument12 - instrument
	satellite3 - satellite
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	image2 - mode
	thermograph1 - mode
	spectrograph0 - mode
	star1 - direction
	star0 - direction
	groundstation2 - direction
	planet3 - direction
	star4 - direction
)
(:init
	(supports instrument0 image2)
	(supports instrument0 thermograph1)
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 star0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet3)
	(supports instrument1 spectrograph0)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 image2)
	(supports instrument2 spectrograph0)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 spectrograph0)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 star0)
	(supports instrument4 image2)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 star0)
	(supports instrument5 spectrograph0)
	(supports instrument5 thermograph1)
	(supports instrument5 image2)
	(calibration_target instrument5 star1)
	(supports instrument6 thermograph1)
	(supports instrument6 image2)
	(calibration_target instrument6 star1)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star4)
	(supports instrument7 spectrograph0)
	(supports instrument7 thermograph1)
	(supports instrument7 image2)
	(calibration_target instrument7 groundstation2)
	(supports instrument8 image2)
	(calibration_target instrument8 star0)
	(supports instrument9 spectrograph0)
	(supports instrument9 image2)
	(supports instrument9 thermograph1)
	(calibration_target instrument9 star0)
	(supports instrument10 thermograph1)
	(supports instrument10 spectrograph0)
	(supports instrument10 image2)
	(calibration_target instrument10 star0)
	(supports instrument11 spectrograph0)
	(supports instrument11 thermograph1)
	(supports instrument11 image2)
	(calibration_target instrument11 groundstation2)
	(supports instrument12 spectrograph0)
	(calibration_target instrument12 star1)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
	(supports instrument13 thermograph1)
	(supports instrument13 image2)
	(calibration_target instrument13 star1)
	(supports instrument14 image2)
	(supports instrument14 thermograph1)
	(calibration_target instrument14 groundstation2)
	(supports instrument15 image2)
	(supports instrument15 thermograph1)
	(supports instrument15 spectrograph0)
	(calibration_target instrument15 star0)
	(supports instrument16 image2)
	(calibration_target instrument16 groundstation2)
	(supports instrument17 spectrograph0)
	(supports instrument17 thermograph1)
	(calibration_target instrument17 groundstation2)
	(supports instrument18 image2)
	(supports instrument18 thermograph1)
	(supports instrument18 spectrograph0)
	(calibration_target instrument18 groundstation2)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(on_board instrument16 satellite3)
	(on_board instrument17 satellite3)
	(on_board instrument18 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation2)
)
(:goal (and
	(pointing satellite2 star0)
	(have_image planet3 thermograph1)
	(have_image star4 thermograph1)
))

)
