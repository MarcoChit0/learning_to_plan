(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	thermograph0 - mode
	infrared1 - mode
	image2 - mode
	image3 - mode
	star0 - direction
	star2 - direction
	star3 - direction
	star1 - direction
	groundstation4 - direction
	phenomenon5 - direction
	planet6 - direction
	phenomenon7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 image2)
	(supports instrument0 thermograph0)
	(supports instrument0 infrared1)
	(calibration_target instrument0 star1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon8)
	(supports instrument1 image2)
	(calibration_target instrument1 groundstation4)
	(supports instrument2 image3)
	(supports instrument2 image2)
	(supports instrument2 infrared1)
	(calibration_target instrument2 groundstation4)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
)
(:goal (and
	(pointing satellite1 star0)
	(have_image phenomenon5 image3)
	(have_image planet6 infrared1)
	(have_image phenomenon7 thermograph0)
	(have_image phenomenon8 infrared1)
))

)
