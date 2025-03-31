(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite1 - satellite
	instrument6 - instrument
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	spectrograph2 - mode
	image1 - mode
	thermograph0 - mode
	groundstation0 - direction
	star1 - direction
)
(:init
	(supports instrument0 thermograph0)
	(supports instrument0 image1)
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 thermograph0)
	(supports instrument2 spectrograph2)
	(supports instrument2 image1)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 image1)
	(supports instrument3 spectrograph2)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 image1)
	(supports instrument4 thermograph0)
	(supports instrument4 spectrograph2)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 image1)
	(calibration_target instrument5 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument6 spectrograph2)
	(supports instrument6 thermograph0)
	(supports instrument6 image1)
	(calibration_target instrument6 groundstation0)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument7 spectrograph2)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 spectrograph2)
	(calibration_target instrument8 groundstation0)
	(supports instrument9 spectrograph2)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 groundstation0)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
	(supports instrument10 thermograph0)
	(supports instrument10 image1)
	(calibration_target instrument10 groundstation0)
	(supports instrument11 spectrograph2)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 groundstation0)
	(supports instrument12 image1)
	(supports instrument12 spectrograph2)
	(supports instrument12 thermograph0)
	(calibration_target instrument12 groundstation0)
	(supports instrument13 spectrograph2)
	(calibration_target instrument13 groundstation0)
	(supports instrument14 spectrograph2)
	(supports instrument14 thermograph0)
	(calibration_target instrument14 groundstation0)
	(supports instrument15 spectrograph2)
	(supports instrument15 image1)
	(supports instrument15 thermograph0)
	(calibration_target instrument15 groundstation0)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation0)
)
(:goal (and
	(pointing satellite2 groundstation0)
	(pointing satellite3 star1)
	(have_image star1 image1)
))

)
