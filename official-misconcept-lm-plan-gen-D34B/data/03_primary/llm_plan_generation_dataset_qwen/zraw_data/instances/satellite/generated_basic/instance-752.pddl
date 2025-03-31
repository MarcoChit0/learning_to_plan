(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	satellite2 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	satellite3 - satellite
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	infrared2 - mode
	spectrograph0 - mode
	thermograph1 - mode
	groundstation1 - direction
	star0 - direction
	groundstation5 - direction
	star2 - direction
	star3 - direction
	star4 - direction
	star6 - direction
	star7 - direction
	phenomenon8 - direction
	phenomenon9 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(supports instrument0 infrared2)
	(supports instrument0 thermograph1)
	(calibration_target instrument0 groundstation5)
	(calibration_target instrument0 star3)
	(supports instrument1 thermograph1)
	(supports instrument1 spectrograph0)
	(supports instrument1 infrared2)
	(calibration_target instrument1 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 star4)
	(supports instrument3 infrared2)
	(calibration_target instrument3 star4)
	(calibration_target instrument3 groundstation5)
	(supports instrument4 infrared2)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 star2)
	(supports instrument5 thermograph1)
	(supports instrument5 infrared2)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 spectrograph0)
	(supports instrument6 infrared2)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 star2)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 star4)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation5)
	(supports instrument8 spectrograph0)
	(supports instrument8 thermograph1)
	(supports instrument8 infrared2)
	(calibration_target instrument8 star3)
	(supports instrument9 infrared2)
	(calibration_target instrument9 star4)
	(supports instrument10 infrared2)
	(supports instrument10 thermograph1)
	(supports instrument10 spectrograph0)
	(calibration_target instrument10 groundstation5)
	(calibration_target instrument10 star0)
	(supports instrument11 infrared2)
	(calibration_target instrument11 star4)
	(calibration_target instrument11 star2)
	(supports instrument12 thermograph1)
	(supports instrument12 infrared2)
	(supports instrument12 spectrograph0)
	(calibration_target instrument12 star2)
	(supports instrument13 spectrograph0)
	(supports instrument13 infrared2)
	(calibration_target instrument13 star0)
	(calibration_target instrument13 groundstation5)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
	(supports instrument14 spectrograph0)
	(supports instrument14 infrared2)
	(calibration_target instrument14 groundstation5)
	(supports instrument15 spectrograph0)
	(supports instrument15 thermograph1)
	(calibration_target instrument15 star4)
	(supports instrument16 spectrograph0)
	(supports instrument16 infrared2)
	(supports instrument16 thermograph1)
	(calibration_target instrument16 star2)
	(supports instrument17 spectrograph0)
	(calibration_target instrument17 star4)
	(calibration_target instrument17 star3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(on_board instrument16 satellite3)
	(on_board instrument17 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star4)
)
(:goal (and
	(pointing satellite2 star0)
	(pointing satellite3 phenomenon8)
	(have_image star6 infrared2)
	(have_image star7 thermograph1)
	(have_image phenomenon8 thermograph1)
	(have_image phenomenon9 spectrograph0)
))

)
