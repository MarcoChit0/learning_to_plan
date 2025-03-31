(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	satellite2 - satellite
	instrument2 - instrument
	instrument3 - instrument
	thermograph2 - mode
	image1 - mode
	infrared0 - mode
	star0 - direction
	groundstation3 - direction
	star5 - direction
	star2 - direction
	star1 - direction
	groundstation4 - direction
	planet6 - direction
	planet7 - direction
	phenomenon8 - direction
	phenomenon9 - direction
)
(:init
	(supports instrument0 thermograph2)
	(calibration_target instrument0 star2)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation3)
	(supports instrument1 thermograph2)
	(supports instrument1 image1)
	(calibration_target instrument1 star1)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument2 infrared0)
	(supports instrument2 image1)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 groundstation4)
	(supports instrument3 image1)
	(calibration_target instrument3 groundstation4)
	(on_board instrument2 satellite2)
	(on_board instrument3 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon9)
)
(:goal (and
	(pointing satellite1 star1)
	(pointing satellite2 star1)
	(have_image planet6 image1)
	(have_image planet7 infrared0)
	(have_image phenomenon8 infrared0)
	(have_image phenomenon9 image1)
))

)
