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
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	satellite3 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	thermograph0 - mode
	spectrograph2 - mode
	image1 - mode
	star0 - direction
	groundstation1 - direction
	star2 - direction
	phenomenon3 - direction
)
(:init
	(supports instrument0 thermograph0)
	(supports instrument0 image1)
	(calibration_target instrument0 star2)
	(supports instrument1 thermograph0)
	(supports instrument1 image1)
	(supports instrument1 spectrograph2)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 image1)
	(calibration_target instrument2 star2)
	(supports instrument3 thermograph0)
	(supports instrument3 spectrograph2)
	(supports instrument3 image1)
	(calibration_target instrument3 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon3)
	(supports instrument4 image1)
	(supports instrument4 thermograph0)
	(supports instrument4 spectrograph2)
	(calibration_target instrument4 groundstation1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument5 image1)
	(supports instrument5 thermograph0)
	(supports instrument5 spectrograph2)
	(calibration_target instrument5 star0)
	(supports instrument6 thermograph0)
	(supports instrument6 spectrograph2)
	(supports instrument6 image1)
	(calibration_target instrument6 star2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
	(supports instrument7 thermograph0)
	(supports instrument7 image1)
	(supports instrument7 spectrograph2)
	(calibration_target instrument7 star0)
	(supports instrument8 thermograph0)
	(supports instrument8 spectrograph2)
	(supports instrument8 image1)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 image1)
	(supports instrument9 spectrograph2)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 star2)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation1)
)
(:goal (and
	(pointing satellite2 groundstation1)
	(pointing satellite3 star0)
	(have_image phenomenon3 spectrograph2)
))

)
