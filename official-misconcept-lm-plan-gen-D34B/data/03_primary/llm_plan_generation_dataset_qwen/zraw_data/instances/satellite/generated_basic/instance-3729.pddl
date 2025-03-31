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
	image2 - mode
	thermograph0 - mode
	infrared1 - mode
	groundstation1 - direction
	star0 - direction
	groundstation5 - direction
	star4 - direction
	groundstation3 - direction
	star2 - direction
	planet6 - direction
)
(:init
	(supports instrument0 image2)
	(calibration_target instrument0 star4)
	(supports instrument1 thermograph0)
	(supports instrument1 image2)
	(supports instrument1 infrared1)
	(calibration_target instrument1 star0)
	(calibration_target instrument1 star2)
	(supports instrument2 image2)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 groundstation5)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation5)
	(supports instrument3 infrared1)
	(supports instrument3 thermograph0)
	(supports instrument3 image2)
	(calibration_target instrument3 star2)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 groundstation3)
	(supports instrument5 thermograph0)
	(supports instrument5 image2)
	(supports instrument5 infrared1)
	(calibration_target instrument5 star4)
	(supports instrument6 infrared1)
	(supports instrument6 thermograph0)
	(supports instrument6 image2)
	(calibration_target instrument6 groundstation3)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation5)
	(supports instrument7 thermograph0)
	(supports instrument7 image2)
	(supports instrument7 infrared1)
	(calibration_target instrument7 star2)
	(calibration_target instrument7 groundstation3)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
)
(:goal (and
	(pointing satellite1 star4)
	(have_image planet6 infrared1)
))

)
