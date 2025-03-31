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
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	thermograph2 - mode
	image3 - mode
	spectrograph0 - mode
	thermograph1 - mode
	star3 - direction
	star2 - direction
	star1 - direction
	groundstation0 - direction
	planet4 - direction
	phenomenon5 - direction
	planet6 - direction
)
(:init
	(supports instrument0 thermograph2)
	(supports instrument0 thermograph1)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 spectrograph0)
	(supports instrument1 thermograph2)
	(supports instrument1 image3)
	(calibration_target instrument1 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument2 image3)
	(calibration_target instrument2 star2)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 star2)
	(supports instrument4 thermograph2)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 image3)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 star2)
	(supports instrument6 spectrograph0)
	(supports instrument6 image3)
	(supports instrument6 thermograph2)
	(calibration_target instrument6 star1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet4)
	(supports instrument7 thermograph1)
	(supports instrument7 thermograph2)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 star2)
	(supports instrument8 image3)
	(supports instrument8 thermograph2)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 star1)
	(supports instrument9 thermograph1)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 star1)
	(supports instrument10 spectrograph0)
	(calibration_target instrument10 star1)
	(supports instrument11 thermograph2)
	(supports instrument11 thermograph1)
	(supports instrument11 spectrograph0)
	(calibration_target instrument11 groundstation0)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet6)
)
(:goal (and
	(have_image planet4 thermograph2)
	(have_image phenomenon5 spectrograph0)
	(have_image planet6 image3)
))

)
