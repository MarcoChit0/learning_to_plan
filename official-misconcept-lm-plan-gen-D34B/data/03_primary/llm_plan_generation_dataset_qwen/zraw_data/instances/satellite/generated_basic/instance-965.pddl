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
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	infrared3 - mode
	thermograph0 - mode
	image1 - mode
	thermograph2 - mode
	groundstation3 - direction
	star2 - direction
	star1 - direction
	groundstation0 - direction
	star4 - direction
	planet5 - direction
	planet6 - direction
)
(:init
	(supports instrument0 thermograph0)
	(supports instrument0 image1)
	(supports instrument0 infrared3)
	(calibration_target instrument0 groundstation3)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument1 image1)
	(supports instrument1 thermograph0)
	(supports instrument1 infrared3)
	(calibration_target instrument1 groundstation3)
	(supports instrument2 thermograph0)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 star1)
	(supports instrument4 infrared3)
	(supports instrument4 thermograph2)
	(calibration_target instrument4 star2)
	(supports instrument5 infrared3)
	(supports instrument5 image1)
	(calibration_target instrument5 star1)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
	(supports instrument6 thermograph2)
	(supports instrument6 infrared3)
	(calibration_target instrument6 groundstation0)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
)
(:goal (and
	(pointing satellite2 planet5)
	(have_image star4 thermograph2)
	(have_image planet5 thermograph0)
	(have_image planet6 thermograph0)
))

)
