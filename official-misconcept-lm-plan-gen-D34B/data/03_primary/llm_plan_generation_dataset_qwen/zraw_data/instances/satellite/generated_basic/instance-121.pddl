(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	instrument4 - instrument
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	spectrograph2 - mode
	spectrograph1 - mode
	infrared0 - mode
	star0 - direction
	star4 - direction
	star5 - direction
	groundstation2 - direction
	groundstation3 - direction
	groundstation1 - direction
	planet6 - direction
	phenomenon7 - direction
	phenomenon8 - direction
	planet9 - direction
	phenomenon10 - direction
	planet11 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 star4)
	(supports instrument2 spectrograph1)
	(supports instrument2 infrared0)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 star5)
	(calibration_target instrument2 star4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation2)
	(supports instrument3 spectrograph2)
	(supports instrument3 spectrograph1)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star5)
	(supports instrument4 spectrograph1)
	(supports instrument4 spectrograph2)
	(calibration_target instrument4 groundstation2)
	(calibration_target instrument4 star5)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet9)
	(supports instrument5 spectrograph2)
	(supports instrument5 infrared0)
	(supports instrument5 spectrograph1)
	(calibration_target instrument5 groundstation1)
	(calibration_target instrument5 groundstation3)
	(supports instrument6 spectrograph2)
	(calibration_target instrument6 groundstation1)
	(calibration_target instrument6 groundstation3)
	(supports instrument7 spectrograph1)
	(calibration_target instrument7 groundstation1)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon10)
)
(:goal (and
	(pointing satellite0 phenomenon8)
	(have_image planet6 spectrograph1)
	(have_image phenomenon7 spectrograph1)
	(have_image phenomenon8 spectrograph2)
	(have_image planet9 spectrograph2)
	(have_image phenomenon10 spectrograph2)
	(have_image planet11 spectrograph1)
))

)
