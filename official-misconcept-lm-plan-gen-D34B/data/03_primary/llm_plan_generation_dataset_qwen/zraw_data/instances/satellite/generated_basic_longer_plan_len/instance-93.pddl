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
	instrument4 - instrument
	instrument5 - instrument
	image0 - mode
	thermograph1 - mode
	image3 - mode
	image2 - mode
	star0 - direction
	star3 - direction
	groundstation1 - direction
	groundstation2 - direction
	phenomenon4 - direction
	planet5 - direction
	phenomenon6 - direction
	planet7 - direction
	star8 - direction
	planet9 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 thermograph1)
	(calibration_target instrument0 groundstation2)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet5)
	(supports instrument1 image2)
	(supports instrument1 thermograph1)
	(supports instrument1 image0)
	(calibration_target instrument1 star3)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet9)
	(supports instrument2 image3)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 thermograph1)
	(supports instrument3 image0)
	(supports instrument3 image3)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 thermograph1)
	(supports instrument4 image3)
	(supports instrument4 image0)
	(calibration_target instrument4 groundstation2)
	(supports instrument5 image2)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation2)
	(on_board instrument2 satellite2)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon4)
)
(:goal (and
	(pointing satellite1 phenomenon4)
	(have_image phenomenon4 image2)
	(have_image planet5 image3)
	(have_image phenomenon6 image2)
	(have_image planet7 image0)
	(have_image star8 image2)
	(have_image planet9 image2)
))

)
