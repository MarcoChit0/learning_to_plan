(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	satellite3 - satellite
	instrument5 - instrument
	instrument6 - instrument
	spectrograph1 - mode
	thermograph0 - mode
	spectrograph2 - mode
	spectrograph3 - mode
	groundstation1 - direction
	groundstation3 - direction
	groundstation0 - direction
	star2 - direction
	star4 - direction
	planet5 - direction
	star6 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(supports instrument0 spectrograph3)
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 groundstation3)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star6)
	(supports instrument1 spectrograph1)
	(supports instrument1 spectrograph2)
	(calibration_target instrument1 star2)
	(supports instrument2 spectrograph2)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 star4)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 groundstation0)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument4 spectrograph1)
	(supports instrument4 spectrograph3)
	(calibration_target instrument4 star2)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star4)
	(supports instrument5 spectrograph3)
	(supports instrument5 thermograph0)
	(supports instrument5 spectrograph1)
	(calibration_target instrument5 star4)
	(supports instrument6 spectrograph2)
	(supports instrument6 spectrograph3)
	(calibration_target instrument6 star4)
	(on_board instrument5 satellite3)
	(on_board instrument6 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star2)
)
(:goal (and
	(pointing satellite3 groundstation0)
	(have_image planet5 spectrograph1)
	(have_image star6 thermograph0)
))

)
