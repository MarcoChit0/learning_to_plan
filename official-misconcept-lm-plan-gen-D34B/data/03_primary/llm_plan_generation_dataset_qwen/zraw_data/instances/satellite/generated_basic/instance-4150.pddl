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
	satellite2 - satellite
	instrument6 - instrument
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
	instrument16 - instrument
	instrument17 - instrument
	infrared2 - mode
	spectrograph1 - mode
	spectrograph3 - mode
	thermograph0 - mode
	star1 - direction
	groundstation0 - direction
	phenomenon2 - direction
)
(:init
	(supports instrument0 spectrograph3)
	(calibration_target instrument0 star1)
	(supports instrument1 infrared2)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 spectrograph1)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 star1)
	(supports instrument4 thermograph0)
	(supports instrument4 infrared2)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 infrared2)
	(supports instrument5 spectrograph3)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 groundstation0)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument6 infrared2)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 star1)
	(supports instrument8 thermograph0)
	(supports instrument8 infrared2)
	(supports instrument8 spectrograph1)
	(calibration_target instrument8 groundstation0)
	(supports instrument9 thermograph0)
	(supports instrument9 infrared2)
	(supports instrument9 spectrograph1)
	(calibration_target instrument9 groundstation0)
	(supports instrument10 infrared2)
	(supports instrument10 spectrograph1)
	(supports instrument10 spectrograph3)
	(calibration_target instrument10 star1)
	(supports instrument11 infrared2)
	(supports instrument11 spectrograph1)
	(calibration_target instrument11 star1)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star1)
	(supports instrument12 thermograph0)
	(calibration_target instrument12 groundstation0)
	(supports instrument13 spectrograph3)
	(calibration_target instrument13 groundstation0)
	(supports instrument14 thermograph0)
	(supports instrument14 spectrograph1)
	(calibration_target instrument14 groundstation0)
	(supports instrument15 spectrograph3)
	(supports instrument15 spectrograph1)
	(calibration_target instrument15 star1)
	(supports instrument16 thermograph0)
	(supports instrument16 spectrograph1)
	(calibration_target instrument16 star1)
	(supports instrument17 thermograph0)
	(calibration_target instrument17 groundstation0)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(on_board instrument16 satellite3)
	(on_board instrument17 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon2)
)
(:goal (and
	(pointing satellite2 phenomenon2)
	(have_image phenomenon2 spectrograph3)
))

)
