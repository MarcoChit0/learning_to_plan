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
	satellite2 - satellite
	instrument6 - instrument
	satellite3 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	satellite4 - satellite
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	infrared1 - mode
	thermograph2 - mode
	spectrograph0 - mode
	spectrograph3 - mode
	groundstation0 - direction
	groundstation1 - direction
	groundstation2 - direction
	star3 - direction
	planet4 - direction
	planet5 - direction
)
(:init
	(supports instrument0 spectrograph3)
	(calibration_target instrument0 star3)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 star3)
	(supports instrument2 spectrograph0)
	(supports instrument2 thermograph2)
	(supports instrument2 spectrograph3)
	(calibration_target instrument2 star3)
	(supports instrument3 spectrograph0)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument4 spectrograph0)
	(supports instrument4 thermograph2)
	(calibration_target instrument4 star3)
	(supports instrument5 thermograph2)
	(supports instrument5 spectrograph0)
	(supports instrument5 spectrograph3)
	(calibration_target instrument5 groundstation1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument6 spectrograph3)
	(calibration_target instrument6 groundstation0)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
	(supports instrument7 thermograph2)
	(supports instrument7 spectrograph0)
	(supports instrument7 infrared1)
	(calibration_target instrument7 star3)
	(supports instrument8 thermograph2)
	(calibration_target instrument8 star3)
	(supports instrument9 thermograph2)
	(supports instrument9 infrared1)
	(calibration_target instrument9 groundstation2)
	(supports instrument10 spectrograph0)
	(supports instrument10 infrared1)
	(supports instrument10 spectrograph3)
	(calibration_target instrument10 groundstation2)
	(supports instrument11 infrared1)
	(supports instrument11 spectrograph0)
	(supports instrument11 spectrograph3)
	(calibration_target instrument11 star3)
	(supports instrument12 thermograph2)
	(calibration_target instrument12 groundstation2)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star3)
	(supports instrument13 spectrograph3)
	(supports instrument13 infrared1)
	(calibration_target instrument13 groundstation0)
	(supports instrument14 infrared1)
	(calibration_target instrument14 groundstation2)
	(supports instrument15 thermograph2)
	(supports instrument15 spectrograph0)
	(supports instrument15 infrared1)
	(calibration_target instrument15 groundstation0)
	(supports instrument16 spectrograph0)
	(calibration_target instrument16 groundstation1)
	(supports instrument17 infrared1)
	(calibration_target instrument17 groundstation2)
	(supports instrument18 spectrograph3)
	(supports instrument18 infrared1)
	(calibration_target instrument18 star3)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation2)
)
(:goal (and
	(pointing satellite0 planet4)
	(pointing satellite2 groundstation0)
	(pointing satellite3 star3)
	(have_image planet4 infrared1)
	(have_image planet5 infrared1)
))

)
