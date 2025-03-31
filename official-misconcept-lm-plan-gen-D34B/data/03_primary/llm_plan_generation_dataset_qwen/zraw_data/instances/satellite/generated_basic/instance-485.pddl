(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	satellite2 - satellite
	instrument2 - instrument
	satellite3 - satellite
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	spectrograph0 - mode
	spectrograph1 - mode
	spectrograph3 - mode
	spectrograph2 - mode
	star2 - direction
	groundstation4 - direction
	groundstation0 - direction
	groundstation1 - direction
	groundstation3 - direction
	planet5 - direction
	star6 - direction
	phenomenon7 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 groundstation0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation4)
	(supports instrument1 spectrograph0)
	(supports instrument1 spectrograph2)
	(calibration_target instrument1 groundstation1)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument2 spectrograph2)
	(supports instrument2 spectrograph3)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 groundstation3)
	(on_board instrument2 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
	(supports instrument3 spectrograph3)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 groundstation3)
	(supports instrument4 spectrograph1)
	(supports instrument4 spectrograph3)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 spectrograph1)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 spectrograph3)
	(calibration_target instrument7 groundstation3)
	(on_board instrument3 satellite3)
	(on_board instrument4 satellite3)
	(on_board instrument5 satellite3)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon7)
)
(:goal (and
	(pointing satellite1 planet5)
	(pointing satellite3 groundstation1)
	(have_image planet5 spectrograph1)
	(have_image star6 spectrograph0)
	(have_image phenomenon7 spectrograph1)
))

)
