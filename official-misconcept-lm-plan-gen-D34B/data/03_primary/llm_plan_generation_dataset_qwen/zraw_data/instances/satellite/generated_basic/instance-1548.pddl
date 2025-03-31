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
	spectrograph0 - mode
	thermograph2 - mode
	image1 - mode
	groundstation0 - direction
	star1 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 spectrograph0)
	(supports instrument0 thermograph2)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 spectrograph0)
	(supports instrument1 image1)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 thermograph2)
	(supports instrument2 image1)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 spectrograph0)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument4 thermograph2)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 spectrograph0)
	(supports instrument5 image1)
	(supports instrument5 thermograph2)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 spectrograph0)
	(supports instrument7 image1)
	(calibration_target instrument7 groundstation0)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument8 thermograph2)
	(supports instrument8 image1)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 groundstation0)
	(supports instrument9 spectrograph0)
	(supports instrument9 thermograph2)
	(calibration_target instrument9 groundstation0)
	(supports instrument10 thermograph2)
	(supports instrument10 spectrograph0)
	(calibration_target instrument10 groundstation0)
	(supports instrument11 thermograph2)
	(supports instrument11 spectrograph0)
	(calibration_target instrument11 groundstation0)
	(supports instrument12 spectrograph0)
	(supports instrument12 thermograph2)
	(calibration_target instrument12 groundstation0)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star1)
	(supports instrument13 spectrograph0)
	(supports instrument13 image1)
	(supports instrument13 thermograph2)
	(calibration_target instrument13 groundstation0)
	(supports instrument14 spectrograph0)
	(supports instrument14 image1)
	(supports instrument14 thermograph2)
	(calibration_target instrument14 groundstation0)
	(supports instrument15 image1)
	(calibration_target instrument15 groundstation0)
	(supports instrument16 thermograph2)
	(supports instrument16 spectrograph0)
	(supports instrument16 image1)
	(calibration_target instrument16 groundstation0)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(on_board instrument16 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star1)
)
(:goal (and
	(pointing satellite2 groundstation0)
	(have_image star1 spectrograph0)
))

)
