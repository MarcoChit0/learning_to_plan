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
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	satellite3 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	satellite4 - satellite
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	spectrograph0 - mode
	spectrograph2 - mode
	thermograph3 - mode
	image1 - mode
	star5 - direction
	groundstation3 - direction
	star1 - direction
	star0 - direction
	star2 - direction
	groundstation4 - direction
	planet6 - direction
	phenomenon7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 spectrograph2)
	(supports instrument0 thermograph3)
	(supports instrument0 image1)
	(calibration_target instrument0 groundstation4)
	(supports instrument1 spectrograph2)
	(supports instrument1 image1)
	(calibration_target instrument1 star5)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation4)
	(supports instrument2 image1)
	(calibration_target instrument2 star0)
	(calibration_target instrument2 star2)
	(supports instrument3 image1)
	(supports instrument3 thermograph3)
	(supports instrument3 spectrograph2)
	(calibration_target instrument3 star5)
	(calibration_target instrument3 star2)
	(supports instrument4 spectrograph2)
	(supports instrument4 thermograph3)
	(calibration_target instrument4 star1)
	(calibration_target instrument4 star0)
	(supports instrument5 thermograph3)
	(supports instrument5 spectrograph0)
	(supports instrument5 image1)
	(calibration_target instrument5 star5)
	(calibration_target instrument5 groundstation3)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
	(supports instrument6 spectrograph2)
	(supports instrument6 image1)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 star0)
	(calibration_target instrument6 star2)
	(supports instrument7 thermograph3)
	(supports instrument7 spectrograph2)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 star1)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon8)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 star2)
	(calibration_target instrument8 star5)
	(supports instrument9 spectrograph0)
	(supports instrument9 thermograph3)
	(supports instrument9 spectrograph2)
	(calibration_target instrument9 star5)
	(supports instrument10 spectrograph2)
	(supports instrument10 image1)
	(calibration_target instrument10 groundstation3)
	(supports instrument11 image1)
	(supports instrument11 spectrograph0)
	(supports instrument11 spectrograph2)
	(calibration_target instrument11 groundstation4)
	(calibration_target instrument11 star0)
	(supports instrument12 thermograph3)
	(supports instrument12 spectrograph0)
	(calibration_target instrument12 groundstation4)
	(calibration_target instrument12 star0)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star2)
	(supports instrument13 spectrograph0)
	(calibration_target instrument13 star2)
	(calibration_target instrument13 star1)
	(supports instrument14 thermograph3)
	(supports instrument14 spectrograph0)
	(supports instrument14 spectrograph2)
	(calibration_target instrument14 star2)
	(calibration_target instrument14 star0)
	(supports instrument15 spectrograph2)
	(supports instrument15 spectrograph0)
	(supports instrument15 thermograph3)
	(calibration_target instrument15 groundstation4)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(power_avail satellite4)
	(pointing satellite4 planet6)
)
(:goal (and
	(pointing satellite2 star2)
	(pointing satellite3 star0)
	(have_image planet6 spectrograph2)
	(have_image phenomenon7 spectrograph2)
	(have_image phenomenon8 spectrograph2)
))

)
