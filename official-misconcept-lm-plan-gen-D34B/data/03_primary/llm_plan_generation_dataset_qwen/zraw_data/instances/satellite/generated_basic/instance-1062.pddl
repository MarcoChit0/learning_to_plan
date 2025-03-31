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
	instrument5 - instrument
	instrument6 - instrument
	infrared1 - mode
	infrared0 - mode
	spectrograph2 - mode
	star5 - direction
	groundstation3 - direction
	star1 - direction
	star0 - direction
	star4 - direction
	star2 - direction
	planet6 - direction
	planet7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 infrared0)
	(supports instrument0 infrared1)
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 spectrograph2)
	(supports instrument1 infrared1)
	(calibration_target instrument1 star2)
	(calibration_target instrument1 star1)
	(supports instrument2 spectrograph2)
	(supports instrument2 infrared1)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation3)
	(calibration_target instrument2 star4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument3 spectrograph2)
	(supports instrument3 infrared1)
	(calibration_target instrument3 groundstation3)
	(calibration_target instrument3 star5)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet7)
	(supports instrument4 infrared0)
	(supports instrument4 spectrograph2)
	(calibration_target instrument4 star0)
	(calibration_target instrument4 star1)
	(supports instrument5 infrared0)
	(supports instrument5 spectrograph2)
	(calibration_target instrument5 star2)
	(calibration_target instrument5 star4)
	(supports instrument6 infrared1)
	(supports instrument6 infrared0)
	(calibration_target instrument6 star2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
)
(:goal (and
	(pointing satellite0 star2)
	(have_image planet6 infrared0)
	(have_image planet7 infrared1)
	(have_image planet8 infrared1)
))

)
