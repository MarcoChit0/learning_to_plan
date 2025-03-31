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
	satellite3 - satellite
	instrument7 - instrument
	instrument8 - instrument
	spectrograph0 - mode
	thermograph2 - mode
	spectrograph3 - mode
	image1 - mode
	star4 - direction
	groundstation3 - direction
	star0 - direction
	groundstation1 - direction
	star2 - direction
	phenomenon5 - direction
	planet6 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 spectrograph3)
	(supports instrument0 thermograph2)
	(calibration_target instrument0 star4)
	(supports instrument1 spectrograph0)
	(supports instrument1 spectrograph3)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument2 thermograph2)
	(supports instrument2 image1)
	(supports instrument2 spectrograph3)
	(calibration_target instrument2 star2)
	(supports instrument3 spectrograph3)
	(supports instrument3 image1)
	(calibration_target instrument3 groundstation3)
	(supports instrument4 image1)
	(calibration_target instrument4 star2)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 star0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star4)
	(supports instrument6 spectrograph0)
	(supports instrument6 spectrograph3)
	(calibration_target instrument6 groundstation1)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon5)
	(supports instrument7 spectrograph3)
	(supports instrument7 thermograph2)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 star2)
	(supports instrument8 image1)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 star2)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation3)
)
(:goal (and
	(pointing satellite0 star4)
	(have_image phenomenon5 thermograph2)
	(have_image planet6 spectrograph3)
))

)
