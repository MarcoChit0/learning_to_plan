(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	satellite3 - satellite
	instrument7 - instrument
	instrument8 - instrument
	spectrograph2 - mode
	spectrograph1 - mode
	spectrograph0 - mode
	groundstation4 - direction
	star3 - direction
	star2 - direction
	star0 - direction
	groundstation1 - direction
	groundstation5 - direction
	phenomenon6 - direction
	star7 - direction
	phenomenon8 - direction
	planet9 - direction
)
(:init
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 star2)
	(calibration_target instrument0 groundstation4)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument1 spectrograph1)
	(supports instrument1 spectrograph0)
	(supports instrument1 spectrograph2)
	(calibration_target instrument1 groundstation5)
	(supports instrument2 spectrograph2)
	(supports instrument2 spectrograph1)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 star3)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 groundstation4)
	(supports instrument4 spectrograph1)
	(calibration_target instrument4 star3)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet9)
	(supports instrument5 spectrograph0)
	(supports instrument5 spectrograph2)
	(supports instrument5 spectrograph1)
	(calibration_target instrument5 star0)
	(supports instrument6 spectrograph1)
	(supports instrument6 spectrograph2)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 star0)
	(calibration_target instrument6 star2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation5)
	(supports instrument7 spectrograph2)
	(supports instrument7 spectrograph1)
	(calibration_target instrument7 groundstation1)
	(calibration_target instrument7 star0)
	(supports instrument8 spectrograph2)
	(supports instrument8 spectrograph1)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 groundstation5)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation5)
)
(:goal (and
	(pointing satellite2 groundstation5)
	(pointing satellite3 star7)
	(have_image phenomenon6 spectrograph0)
	(have_image star7 spectrograph1)
	(have_image phenomenon8 spectrograph1)
	(have_image planet9 spectrograph2)
))

)
