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
	infrared1 - mode
	spectrograph2 - mode
	infrared0 - mode
	star0 - direction
	planet1 - direction
	star2 - direction
)
(:init
	(supports instrument0 spectrograph2)
	(supports instrument0 infrared1)
	(supports instrument0 infrared0)
	(calibration_target instrument0 star0)
	(supports instrument1 spectrograph2)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star0)
	(supports instrument2 infrared0)
	(supports instrument2 spectrograph2)
	(supports instrument2 infrared1)
	(calibration_target instrument2 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument3 spectrograph2)
	(supports instrument3 infrared0)
	(supports instrument3 infrared1)
	(calibration_target instrument3 star0)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet1)
	(supports instrument4 infrared1)
	(supports instrument4 spectrograph2)
	(supports instrument4 infrared0)
	(calibration_target instrument4 star0)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet1)
	(supports instrument5 infrared1)
	(supports instrument5 spectrograph2)
	(calibration_target instrument5 star0)
	(supports instrument6 infrared1)
	(supports instrument6 spectrograph2)
	(supports instrument6 infrared0)
	(calibration_target instrument6 star0)
	(supports instrument7 infrared1)
	(supports instrument7 infrared0)
	(calibration_target instrument7 star0)
	(on_board instrument5 satellite3)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet1)
)
(:goal (and
	(pointing satellite0 star2)
	(pointing satellite1 star2)
	(have_image planet1 infrared1)
	(have_image star2 spectrograph2)
))

)
