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
	instrument5 - instrument
	instrument6 - instrument
	image2 - mode
	spectrograph0 - mode
	spectrograph1 - mode
	infrared3 - mode
	groundstation1 - direction
	star2 - direction
	groundstation0 - direction
	planet3 - direction
	phenomenon4 - direction
	planet5 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 star2)
	(supports instrument1 infrared3)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 spectrograph0)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument3 infrared3)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 star2)
	(supports instrument4 infrared3)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 image2)
	(calibration_target instrument5 star2)
	(supports instrument6 infrared3)
	(supports instrument6 spectrograph1)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 groundstation0)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet3)
)
(:goal (and
	(have_image planet3 infrared3)
	(have_image phenomenon4 infrared3)
	(have_image planet5 spectrograph1)
))

)
