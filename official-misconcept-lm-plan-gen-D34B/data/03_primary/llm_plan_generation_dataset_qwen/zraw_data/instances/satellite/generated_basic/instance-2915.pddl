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
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	satellite3 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	satellite4 - satellite
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	spectrograph2 - mode
	infrared3 - mode
	thermograph1 - mode
	thermograph0 - mode
	groundstation0 - direction
	star1 - direction
	planet2 - direction
	star3 - direction
)
(:init
	(supports instrument0 thermograph0)
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 star1)
	(supports instrument1 thermograph1)
	(supports instrument1 thermograph0)
	(supports instrument1 infrared3)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 infrared3)
	(calibration_target instrument2 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star1)
	(supports instrument4 spectrograph2)
	(supports instrument4 thermograph1)
	(supports instrument4 infrared3)
	(calibration_target instrument4 groundstation0)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
	(supports instrument5 thermograph0)
	(supports instrument5 spectrograph2)
	(supports instrument5 infrared3)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 spectrograph2)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 thermograph1)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 star1)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 groundstation0)
	(supports instrument10 spectrograph2)
	(calibration_target instrument10 groundstation0)
	(supports instrument11 thermograph1)
	(supports instrument11 infrared3)
	(calibration_target instrument11 star1)
	(supports instrument12 thermograph0)
	(supports instrument12 spectrograph2)
	(supports instrument12 thermograph1)
	(calibration_target instrument12 groundstation0)
	(supports instrument13 thermograph0)
	(calibration_target instrument13 groundstation0)
	(supports instrument14 thermograph0)
	(supports instrument14 infrared3)
	(supports instrument14 thermograph1)
	(calibration_target instrument14 star1)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet2)
	(supports instrument15 thermograph1)
	(calibration_target instrument15 star1)
	(supports instrument16 thermograph1)
	(supports instrument16 infrared3)
	(supports instrument16 thermograph0)
	(calibration_target instrument16 groundstation0)
	(supports instrument17 spectrograph2)
	(calibration_target instrument17 star1)
	(supports instrument18 thermograph0)
	(supports instrument18 infrared3)
	(calibration_target instrument18 groundstation0)
	(supports instrument19 thermograph0)
	(supports instrument19 spectrograph2)
	(calibration_target instrument19 star1)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(on_board instrument19 satellite4)
	(power_avail satellite4)
	(pointing satellite4 planet2)
)
(:goal (and
	(pointing satellite1 groundstation0)
	(have_image planet2 infrared3)
	(have_image star3 thermograph1)
))

)
