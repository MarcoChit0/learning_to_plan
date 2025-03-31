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
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	satellite4 - satellite
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	thermograph0 - mode
	thermograph2 - mode
	infrared3 - mode
	spectrograph1 - mode
	groundstation0 - direction
	star2 - direction
	star1 - direction
	star3 - direction
	phenomenon4 - direction
	phenomenon5 - direction
	phenomenon6 - direction
	star7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 thermograph2)
	(supports instrument1 thermograph0)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 star1)
	(supports instrument2 thermograph0)
	(supports instrument2 spectrograph1)
	(supports instrument2 infrared3)
	(calibration_target instrument2 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 infrared3)
	(supports instrument4 thermograph2)
	(supports instrument4 spectrograph1)
	(calibration_target instrument4 star2)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument5 infrared3)
	(supports instrument5 spectrograph1)
	(supports instrument5 thermograph2)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 spectrograph1)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 star2)
	(supports instrument8 spectrograph1)
	(supports instrument8 infrared3)
	(calibration_target instrument8 star2)
	(supports instrument9 infrared3)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 star2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
	(supports instrument10 thermograph2)
	(supports instrument10 thermograph0)
	(calibration_target instrument10 groundstation0)
	(supports instrument11 infrared3)
	(calibration_target instrument11 star1)
	(supports instrument12 thermograph2)
	(supports instrument12 thermograph0)
	(calibration_target instrument12 star1)
	(supports instrument13 infrared3)
	(supports instrument13 spectrograph1)
	(calibration_target instrument13 star2)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon6)
	(supports instrument14 thermograph0)
	(supports instrument14 spectrograph1)
	(supports instrument14 thermograph2)
	(calibration_target instrument14 groundstation0)
	(supports instrument15 thermograph2)
	(calibration_target instrument15 star2)
	(supports instrument16 thermograph2)
	(supports instrument16 thermograph0)
	(supports instrument16 infrared3)
	(calibration_target instrument16 star1)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star2)
)
(:goal (and
	(pointing satellite1 star2)
	(pointing satellite2 phenomenon4)
	(pointing satellite3 star1)
	(pointing satellite4 star3)
	(have_image star3 thermograph2)
	(have_image phenomenon4 thermograph0)
	(have_image phenomenon5 infrared3)
	(have_image phenomenon6 thermograph0)
	(have_image star7 thermograph0)
	(have_image phenomenon8 thermograph2)
))

)
