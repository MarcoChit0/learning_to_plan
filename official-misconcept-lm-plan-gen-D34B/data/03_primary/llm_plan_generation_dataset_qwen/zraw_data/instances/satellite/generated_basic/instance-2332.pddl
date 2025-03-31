(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	thermograph1 - mode
	thermograph0 - mode
	image2 - mode
	groundstation0 - direction
	star4 - direction
	groundstation2 - direction
	star3 - direction
	groundstation1 - direction
	phenomenon5 - direction
	planet6 - direction
	phenomenon7 - direction
	star8 - direction
	phenomenon9 - direction
)
(:init
	(supports instrument0 thermograph0)
	(supports instrument0 image2)
	(supports instrument0 thermograph1)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 image2)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon7)
	(supports instrument2 image2)
	(supports instrument2 thermograph1)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 groundstation1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon5)
)
(:goal (and
	(pointing satellite0 groundstation2)
	(pointing satellite1 star4)
	(have_image phenomenon5 image2)
	(have_image planet6 image2)
	(have_image phenomenon7 image2)
	(have_image star8 thermograph1)
	(have_image phenomenon9 thermograph1)
))

)
