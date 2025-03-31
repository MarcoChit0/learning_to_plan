(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	satellite3 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	satellite4 - satellite
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	spectrograph2 - mode
	thermograph1 - mode
	image0 - mode
	thermograph3 - mode
	star3 - direction
	groundstation0 - direction
	star1 - direction
	groundstation2 - direction
	planet4 - direction
	phenomenon5 - direction
)
(:init
	(supports instrument0 spectrograph2)
	(supports instrument0 thermograph3)
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 thermograph1)
	(supports instrument2 image0)
	(supports instrument2 thermograph3)
	(calibration_target instrument2 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon5)
	(supports instrument3 spectrograph2)
	(supports instrument3 thermograph3)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 star1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 image0)
	(supports instrument6 thermograph3)
	(supports instrument6 spectrograph2)
	(calibration_target instrument6 groundstation2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
	(supports instrument7 thermograph3)
	(supports instrument7 thermograph1)
	(supports instrument7 spectrograph2)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 thermograph3)
	(supports instrument8 spectrograph2)
	(supports instrument8 image0)
	(calibration_target instrument8 groundstation2)
	(supports instrument9 spectrograph2)
	(supports instrument9 image0)
	(supports instrument9 thermograph3)
	(calibration_target instrument9 groundstation2)
	(supports instrument10 spectrograph2)
	(supports instrument10 thermograph1)
	(calibration_target instrument10 groundstation2)
	(supports instrument11 image0)
	(supports instrument11 thermograph3)
	(supports instrument11 spectrograph2)
	(calibration_target instrument11 groundstation2)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet4)
	(supports instrument12 spectrograph2)
	(supports instrument12 thermograph1)
	(supports instrument12 thermograph3)
	(calibration_target instrument12 groundstation2)
	(supports instrument13 thermograph1)
	(supports instrument13 thermograph3)
	(supports instrument13 spectrograph2)
	(calibration_target instrument13 star1)
	(supports instrument14 spectrograph2)
	(supports instrument14 thermograph1)
	(calibration_target instrument14 groundstation2)
	(supports instrument15 spectrograph2)
	(calibration_target instrument15 groundstation2)
	(on_board instrument12 satellite4)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation0)
)
(:goal (and
	(pointing satellite4 groundstation2)
	(have_image planet4 spectrograph2)
	(have_image phenomenon5 thermograph3)
))

)
