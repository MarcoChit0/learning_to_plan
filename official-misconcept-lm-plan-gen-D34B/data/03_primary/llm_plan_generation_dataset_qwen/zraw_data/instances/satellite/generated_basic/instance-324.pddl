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
	thermograph2 - mode
	spectrograph3 - mode
	infrared0 - mode
	infrared1 - mode
	star2 - direction
	groundstation5 - direction
	star4 - direction
	groundstation0 - direction
	groundstation3 - direction
	star1 - direction
	planet6 - direction
)
(:init
	(supports instrument0 thermograph2)
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation0)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 infrared1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star4)
	(calibration_target instrument1 groundstation3)
	(supports instrument2 spectrograph3)
	(supports instrument2 infrared1)
	(calibration_target instrument2 star1)
	(calibration_target instrument2 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet6)
	(supports instrument3 infrared1)
	(calibration_target instrument3 star2)
	(supports instrument4 infrared0)
	(supports instrument4 infrared1)
	(supports instrument4 thermograph2)
	(calibration_target instrument4 star4)
	(calibration_target instrument4 groundstation5)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
	(supports instrument5 infrared0)
	(supports instrument5 infrared1)
	(supports instrument5 spectrograph3)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 infrared1)
	(calibration_target instrument6 star1)
	(calibration_target instrument6 groundstation3)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
)
(:goal (and
	(pointing satellite0 groundstation3)
	(pointing satellite2 star1)
	(have_image planet6 infrared0)
))

)
