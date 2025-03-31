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
	image1 - mode
	thermograph2 - mode
	image0 - mode
	star3 - direction
	groundstation5 - direction
	groundstation2 - direction
	star1 - direction
	star0 - direction
	groundstation4 - direction
	planet6 - direction
	planet7 - direction
	phenomenon8 - direction
	phenomenon9 - direction
	phenomenon10 - direction
)
(:init
	(supports instrument0 thermograph2)
	(supports instrument0 image0)
	(supports instrument0 image1)
	(calibration_target instrument0 groundstation5)
	(supports instrument1 image1)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 groundstation5)
	(supports instrument2 image1)
	(supports instrument2 thermograph2)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation2)
	(calibration_target instrument2 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument3 image1)
	(calibration_target instrument3 groundstation2)
	(calibration_target instrument3 groundstation5)
	(supports instrument4 thermograph2)
	(supports instrument4 image1)
	(calibration_target instrument4 star0)
	(calibration_target instrument4 star1)
	(supports instrument5 image1)
	(supports instrument5 thermograph2)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation4)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet7)
)
(:goal (and
	(pointing satellite0 phenomenon10)
	(pointing satellite1 star3)
	(have_image planet6 image1)
	(have_image planet7 thermograph2)
	(have_image phenomenon8 thermograph2)
	(have_image phenomenon9 image0)
	(have_image phenomenon10 thermograph2)
))

)
