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
	satellite2 - satellite
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
	spectrograph0 - mode
	infrared1 - mode
	thermograph2 - mode
	groundstation0 - direction
	star1 - direction
	groundstation2 - direction
	phenomenon3 - direction
	star4 - direction
	star5 - direction
	star6 - direction
	star7 - direction
	star8 - direction
)
(:init
	(supports instrument0 thermograph2)
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 infrared1)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 infrared1)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon3)
	(supports instrument3 thermograph2)
	(supports instrument3 infrared1)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 star1)
	(supports instrument5 thermograph2)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 thermograph2)
	(calibration_target instrument7 groundstation2)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star7)
	(supports instrument8 infrared1)
	(supports instrument8 thermograph2)
	(calibration_target instrument8 star1)
	(supports instrument9 thermograph2)
	(supports instrument9 spectrograph0)
	(supports instrument9 infrared1)
	(calibration_target instrument9 star1)
	(supports instrument10 infrared1)
	(supports instrument10 thermograph2)
	(supports instrument10 spectrograph0)
	(calibration_target instrument10 groundstation2)
	(supports instrument11 infrared1)
	(calibration_target instrument11 groundstation2)
	(supports instrument12 spectrograph0)
	(supports instrument12 thermograph2)
	(supports instrument12 infrared1)
	(calibration_target instrument12 groundstation0)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star7)
	(supports instrument13 thermograph2)
	(supports instrument13 spectrograph0)
	(calibration_target instrument13 groundstation2)
	(supports instrument14 infrared1)
	(calibration_target instrument14 groundstation2)
	(supports instrument15 spectrograph0)
	(calibration_target instrument15 star1)
	(supports instrument16 spectrograph0)
	(supports instrument16 thermograph2)
	(supports instrument16 infrared1)
	(calibration_target instrument16 star1)
	(supports instrument17 spectrograph0)
	(supports instrument17 thermograph2)
	(supports instrument17 infrared1)
	(calibration_target instrument17 groundstation2)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(on_board instrument16 satellite3)
	(on_board instrument17 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star8)
)
(:goal (and
	(pointing satellite0 star8)
	(pointing satellite3 star4)
	(have_image phenomenon3 infrared1)
	(have_image star4 spectrograph0)
	(have_image star5 infrared1)
	(have_image star6 spectrograph0)
	(have_image star7 spectrograph0)
	(have_image star8 infrared1)
))

)
