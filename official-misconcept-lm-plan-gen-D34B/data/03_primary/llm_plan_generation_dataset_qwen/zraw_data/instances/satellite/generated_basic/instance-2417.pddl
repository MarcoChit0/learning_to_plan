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
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	satellite3 - satellite
	instrument8 - instrument
	spectrograph1 - mode
	spectrograph3 - mode
	spectrograph2 - mode
	infrared0 - mode
	groundstation1 - direction
	groundstation3 - direction
	groundstation2 - direction
	groundstation0 - direction
	groundstation4 - direction
	planet5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 spectrograph3)
	(calibration_target instrument0 groundstation4)
	(supports instrument1 spectrograph3)
	(calibration_target instrument1 groundstation4)
	(supports instrument2 spectrograph2)
	(supports instrument2 infrared0)
	(supports instrument2 spectrograph3)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 spectrograph3)
	(calibration_target instrument3 groundstation4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon6)
	(supports instrument4 spectrograph2)
	(supports instrument4 infrared0)
	(supports instrument4 spectrograph3)
	(calibration_target instrument4 groundstation3)
	(supports instrument5 spectrograph3)
	(supports instrument5 spectrograph2)
	(calibration_target instrument5 groundstation4)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument6 spectrograph3)
	(supports instrument6 spectrograph2)
	(calibration_target instrument6 groundstation2)
	(supports instrument7 spectrograph3)
	(calibration_target instrument7 groundstation0)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation3)
	(supports instrument8 spectrograph3)
	(supports instrument8 spectrograph1)
	(calibration_target instrument8 groundstation4)
	(on_board instrument8 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation0)
)
(:goal (and
	(pointing satellite1 planet5)
	(pointing satellite2 planet5)
	(pointing satellite3 planet5)
	(have_image planet5 spectrograph1)
	(have_image phenomenon6 spectrograph3)
))

)
