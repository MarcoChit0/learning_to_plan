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
	satellite3 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	spectrograph0 - mode
	spectrograph2 - mode
	spectrograph1 - mode
	thermograph3 - mode
	groundstation1 - direction
	groundstation3 - direction
	star4 - direction
	star2 - direction
	groundstation0 - direction
	phenomenon5 - direction
	star6 - direction
	planet7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 thermograph3)
	(supports instrument0 spectrograph2)
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 star4)
	(supports instrument1 spectrograph0)
	(supports instrument1 thermograph3)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 spectrograph1)
	(supports instrument2 spectrograph2)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation3)
	(supports instrument3 spectrograph1)
	(supports instrument3 spectrograph0)
	(supports instrument3 thermograph3)
	(calibration_target instrument3 star4)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet8)
	(supports instrument4 thermograph3)
	(supports instrument4 spectrograph0)
	(supports instrument4 spectrograph2)
	(calibration_target instrument4 groundstation0)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon5)
	(supports instrument5 spectrograph1)
	(supports instrument5 spectrograph0)
	(supports instrument5 thermograph3)
	(calibration_target instrument5 star2)
	(supports instrument6 spectrograph1)
	(supports instrument6 thermograph3)
	(supports instrument6 spectrograph2)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 groundstation0)
	(on_board instrument5 satellite3)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon5)
)
(:goal (and
	(pointing satellite1 star2)
	(pointing satellite2 groundstation0)
	(have_image phenomenon5 spectrograph2)
	(have_image star6 spectrograph2)
	(have_image planet7 thermograph3)
	(have_image planet8 thermograph3)
))

)
