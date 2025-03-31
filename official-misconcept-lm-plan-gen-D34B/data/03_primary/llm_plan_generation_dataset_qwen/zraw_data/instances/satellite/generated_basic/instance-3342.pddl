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
	infrared1 - mode
	infrared0 - mode
	thermograph2 - mode
	thermograph3 - mode
	star0 - direction
	groundstation1 - direction
	planet2 - direction
	planet3 - direction
)
(:init
	(supports instrument0 thermograph3)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument3 thermograph3)
	(supports instrument3 infrared1)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument4 infrared1)
	(calibration_target instrument4 groundstation1)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet2)
)
(:goal (and
	(pointing satellite0 planet2)
	(pointing satellite1 groundstation1)
	(pointing satellite2 planet3)
	(have_image planet2 infrared1)
	(have_image planet3 infrared1)
))

)
