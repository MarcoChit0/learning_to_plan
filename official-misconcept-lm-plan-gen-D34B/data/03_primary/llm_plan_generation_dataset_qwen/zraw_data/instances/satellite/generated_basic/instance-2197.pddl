(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	satellite2 - satellite
	instrument3 - instrument
	image2 - mode
	image0 - mode
	infrared3 - mode
	thermograph1 - mode
	star0 - direction
	star3 - direction
	star1 - direction
	groundstation2 - direction
	star4 - direction
	phenomenon5 - direction
	phenomenon6 - direction
	star7 - direction
	phenomenon8 - direction
	phenomenon9 - direction
	star10 - direction
)
(:init
	(supports instrument0 infrared3)
	(supports instrument0 image2)
	(supports instrument0 image0)
	(calibration_target instrument0 star4)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon9)
	(supports instrument1 thermograph1)
	(supports instrument1 image2)
	(supports instrument1 image0)
	(calibration_target instrument1 star1)
	(supports instrument2 image2)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation2)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star7)
	(supports instrument3 image2)
	(calibration_target instrument3 star4)
	(on_board instrument3 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation2)
)
(:goal (and
	(pointing satellite1 phenomenon5)
	(have_image phenomenon5 infrared3)
	(have_image phenomenon6 image0)
	(have_image star7 infrared3)
	(have_image phenomenon8 image2)
	(have_image phenomenon9 image2)
	(have_image star10 image2)
))

)
