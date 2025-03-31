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
	satellite2 - satellite
	instrument7 - instrument
	infrared2 - mode
	spectrograph1 - mode
	infrared0 - mode
	groundstation0 - direction
	star1 - direction
	planet2 - direction
	phenomenon3 - direction
	phenomenon4 - direction
)
(:init
	(supports instrument0 infrared2)
	(calibration_target instrument0 star1)
	(supports instrument1 infrared0)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 spectrograph1)
	(supports instrument2 infrared0)
	(supports instrument2 infrared2)
	(calibration_target instrument2 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon4)
	(supports instrument3 spectrograph1)
	(supports instrument3 infrared2)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 spectrograph1)
	(supports instrument4 infrared0)
	(supports instrument4 infrared2)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 infrared0)
	(supports instrument5 infrared2)
	(supports instrument5 spectrograph1)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 infrared0)
	(calibration_target instrument6 star1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon4)
	(supports instrument7 spectrograph1)
	(supports instrument7 infrared2)
	(supports instrument7 infrared0)
	(calibration_target instrument7 star1)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet2)
)
(:goal (and
	(pointing satellite0 planet2)
	(pointing satellite2 phenomenon3)
	(have_image planet2 infrared2)
	(have_image phenomenon3 infrared2)
	(have_image phenomenon4 infrared0)
))

)
