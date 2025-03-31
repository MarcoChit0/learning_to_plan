(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite1 - satellite
	instrument5 - instrument
	instrument6 - instrument
	spectrograph1 - mode
	infrared0 - mode
	groundstation0 - direction
	groundstation3 - direction
	star5 - direction
	groundstation1 - direction
	groundstation4 - direction
	star2 - direction
	phenomenon6 - direction
	planet7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 groundstation4)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star2)
	(supports instrument2 spectrograph1)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 groundstation3)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 spectrograph1)
	(supports instrument4 infrared0)
	(calibration_target instrument4 star5)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon6)
	(supports instrument5 infrared0)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 infrared0)
	(calibration_target instrument6 star2)
	(calibration_target instrument6 groundstation4)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation4)
)
(:goal (and
	(pointing satellite0 star2)
	(pointing satellite1 planet7)
	(have_image phenomenon6 infrared0)
	(have_image planet7 spectrograph1)
	(have_image planet8 infrared0)
))

)
