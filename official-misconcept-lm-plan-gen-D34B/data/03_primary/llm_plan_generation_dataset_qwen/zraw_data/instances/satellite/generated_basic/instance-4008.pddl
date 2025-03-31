(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	satellite3 - satellite
	instrument5 - instrument
	image0 - mode
	image1 - mode
	image2 - mode
	groundstation1 - direction
	groundstation2 - direction
	star0 - direction
	phenomenon3 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 image2)
	(supports instrument0 image0)
	(calibration_target instrument0 star0)
	(supports instrument1 image2)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation2)
	(supports instrument2 image2)
	(supports instrument2 image1)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation2)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon3)
	(supports instrument4 image2)
	(supports instrument4 image1)
	(calibration_target instrument4 groundstation2)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon3)
	(supports instrument5 image0)
	(supports instrument5 image1)
	(calibration_target instrument5 star0)
	(on_board instrument5 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star0)
)
(:goal (and
	(pointing satellite2 star0)
	(pointing satellite3 phenomenon3)
	(have_image phenomenon3 image1)
))

)
