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
	infrared0 - mode
	thermograph2 - mode
	image1 - mode
	groundstation1 - direction
	star0 - direction
	planet2 - direction
	phenomenon3 - direction
	star4 - direction
	phenomenon5 - direction
	planet6 - direction
)
(:init
	(supports instrument0 infrared0)
	(supports instrument0 thermograph2)
	(supports instrument0 image1)
	(calibration_target instrument0 star0)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 infrared0)
	(supports instrument2 image1)
	(calibration_target instrument2 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet6)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument5 infrared0)
	(calibration_target instrument5 star0)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
)
(:goal (and
	(pointing satellite1 planet6)
	(pointing satellite2 planet2)
	(have_image planet2 thermograph2)
	(have_image phenomenon3 image1)
	(have_image star4 image1)
	(have_image phenomenon5 image1)
	(have_image planet6 thermograph2)
))

)
