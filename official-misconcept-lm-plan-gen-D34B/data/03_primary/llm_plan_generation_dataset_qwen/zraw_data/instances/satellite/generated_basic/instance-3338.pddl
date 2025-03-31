(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
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
	instrument14 - instrument
	image1 - mode
	spectrograph0 - mode
	thermograph2 - mode
	star0 - direction
	star1 - direction
	groundstation2 - direction
	phenomenon3 - direction
	planet4 - direction
)
(:init
	(supports instrument0 thermograph2)
	(supports instrument0 spectrograph0)
	(supports instrument0 image1)
	(calibration_target instrument0 star1)
	(supports instrument1 spectrograph0)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet4)
	(supports instrument2 thermograph2)
	(supports instrument2 image1)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 star1)
	(supports instrument3 image1)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 star0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 groundstation2)
	(supports instrument5 image1)
	(supports instrument5 spectrograph0)
	(supports instrument5 thermograph2)
	(calibration_target instrument5 groundstation2)
	(supports instrument6 thermograph2)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 star1)
	(supports instrument7 thermograph2)
	(supports instrument7 image1)
	(calibration_target instrument7 star1)
	(supports instrument8 image1)
	(calibration_target instrument8 groundstation2)
	(supports instrument9 spectrograph0)
	(supports instrument9 thermograph2)
	(supports instrument9 image1)
	(calibration_target instrument9 star0)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet4)
	(supports instrument10 thermograph2)
	(supports instrument10 image1)
	(supports instrument10 spectrograph0)
	(calibration_target instrument10 groundstation2)
	(supports instrument11 image1)
	(supports instrument11 spectrograph0)
	(calibration_target instrument11 groundstation2)
	(supports instrument12 spectrograph0)
	(calibration_target instrument12 star1)
	(supports instrument13 thermograph2)
	(supports instrument13 spectrograph0)
	(calibration_target instrument13 groundstation2)
	(supports instrument14 image1)
	(supports instrument14 thermograph2)
	(supports instrument14 spectrograph0)
	(calibration_target instrument14 groundstation2)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet4)
)
(:goal (and
	(pointing satellite2 phenomenon3)
	(have_image phenomenon3 image1)
	(have_image planet4 thermograph2)
))

)
