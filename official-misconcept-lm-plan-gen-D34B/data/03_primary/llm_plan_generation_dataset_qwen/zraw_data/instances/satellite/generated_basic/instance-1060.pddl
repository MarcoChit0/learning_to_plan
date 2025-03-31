(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	satellite2 - satellite
	instrument3 - instrument
	image2 - mode
	image0 - mode
	infrared1 - mode
	groundstation0 - direction
	groundstation5 - direction
	groundstation4 - direction
	star2 - direction
	groundstation3 - direction
	star1 - direction
	phenomenon6 - direction
	phenomenon7 - direction
	star8 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 image2)
	(supports instrument0 infrared1)
	(calibration_target instrument0 groundstation5)
	(calibration_target instrument0 star2)
	(supports instrument1 image2)
	(supports instrument1 infrared1)
	(calibration_target instrument1 star2)
	(calibration_target instrument1 groundstation4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument2 image2)
	(calibration_target instrument2 star1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument3 image0)
	(supports instrument3 image2)
	(calibration_target instrument3 star1)
	(calibration_target instrument3 groundstation3)
	(on_board instrument3 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation4)
)
(:goal (and
	(pointing satellite0 groundstation5)
	(have_image phenomenon6 image2)
	(have_image phenomenon7 image2)
	(have_image star8 infrared1)
))

)
