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
	satellite2 - satellite
	instrument6 - instrument
	spectrograph2 - mode
	infrared1 - mode
	infrared0 - mode
	groundstation4 - direction
	groundstation5 - direction
	star3 - direction
	star2 - direction
	star1 - direction
	star0 - direction
	planet6 - direction
)
(:init
	(supports instrument0 spectrograph2)
	(supports instrument0 infrared1)
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation4)
	(supports instrument1 infrared0)
	(supports instrument1 infrared1)
	(calibration_target instrument1 star3)
	(supports instrument2 infrared1)
	(supports instrument2 infrared0)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 star2)
	(calibration_target instrument2 groundstation5)
	(supports instrument3 spectrograph2)
	(calibration_target instrument3 star3)
	(supports instrument4 infrared0)
	(calibration_target instrument4 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument5 infrared1)
	(supports instrument5 spectrograph2)
	(supports instrument5 infrared0)
	(calibration_target instrument5 star2)
	(calibration_target instrument5 star0)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
	(supports instrument6 spectrograph2)
	(supports instrument6 infrared1)
	(calibration_target instrument6 star0)
	(calibration_target instrument6 star1)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet6)
)
(:goal (and
	(pointing satellite2 planet6)
	(have_image planet6 infrared0)
))

)
