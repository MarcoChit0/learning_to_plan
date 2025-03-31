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
	instrument7 - instrument
	satellite2 - satellite
	instrument8 - instrument
	image1 - mode
	thermograph0 - mode
	star4 - direction
	star0 - direction
	groundstation5 - direction
	groundstation3 - direction
	groundstation1 - direction
	star2 - direction
	planet6 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star4)
	(calibration_target instrument0 star0)
	(supports instrument1 thermograph0)
	(supports instrument1 image1)
	(calibration_target instrument1 star0)
	(supports instrument2 image1)
	(calibration_target instrument2 groundstation3)
	(calibration_target instrument2 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 groundstation3)
	(supports instrument4 image1)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 star2)
	(supports instrument5 thermograph0)
	(supports instrument5 image1)
	(calibration_target instrument5 groundstation5)
	(supports instrument6 image1)
	(calibration_target instrument6 groundstation3)
	(supports instrument7 image1)
	(calibration_target instrument7 groundstation1)
	(calibration_target instrument7 groundstation3)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation5)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 star2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star4)
)
(:goal (and
	(pointing satellite2 groundstation3)
	(have_image planet6 thermograph0)
))

)
