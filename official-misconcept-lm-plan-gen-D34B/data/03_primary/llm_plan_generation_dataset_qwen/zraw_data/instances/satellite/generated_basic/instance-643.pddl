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
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite3 - satellite
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
	instrument20 - instrument
	instrument21 - instrument
	spectrograph1 - mode
	thermograph0 - mode
	image2 - mode
	groundstation2 - direction
	star0 - direction
	star3 - direction
	groundstation1 - direction
	planet4 - direction
	planet5 - direction
	planet6 - direction
)
(:init
	(supports instrument0 thermograph0)
	(supports instrument0 image2)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 image2)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 star0)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 star3)
	(supports instrument4 spectrograph1)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 groundstation2)
	(supports instrument5 image2)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 groundstation2)
	(supports instrument6 image2)
	(calibration_target instrument6 star3)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
	(supports instrument7 spectrograph1)
	(supports instrument7 image2)
	(calibration_target instrument7 groundstation2)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 groundstation2)
	(supports instrument9 thermograph0)
	(supports instrument9 spectrograph1)
	(supports instrument9 image2)
	(calibration_target instrument9 groundstation2)
	(supports instrument10 thermograph0)
	(supports instrument10 spectrograph1)
	(supports instrument10 image2)
	(calibration_target instrument10 groundstation1)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 groundstation1)
	(supports instrument12 thermograph0)
	(supports instrument12 spectrograph1)
	(calibration_target instrument12 star0)
	(supports instrument13 image2)
	(calibration_target instrument13 groundstation2)
	(supports instrument14 spectrograph1)
	(supports instrument14 image2)
	(supports instrument14 thermograph0)
	(calibration_target instrument14 groundstation1)
	(supports instrument15 thermograph0)
	(supports instrument15 spectrograph1)
	(calibration_target instrument15 groundstation2)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet4)
	(supports instrument16 image2)
	(supports instrument16 thermograph0)
	(calibration_target instrument16 star0)
	(supports instrument17 spectrograph1)
	(calibration_target instrument17 groundstation1)
	(supports instrument18 spectrograph1)
	(calibration_target instrument18 star3)
	(supports instrument19 image2)
	(supports instrument19 spectrograph1)
	(calibration_target instrument19 star3)
	(supports instrument20 spectrograph1)
	(calibration_target instrument20 groundstation1)
	(supports instrument21 image2)
	(supports instrument21 spectrograph1)
	(supports instrument21 thermograph0)
	(calibration_target instrument21 groundstation1)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(on_board instrument19 satellite4)
	(on_board instrument20 satellite4)
	(on_board instrument21 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star0)
)
(:goal (and
	(have_image planet4 thermograph0)
	(have_image planet5 spectrograph1)
	(have_image planet6 thermograph0)
))

)
