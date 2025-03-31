(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite1 - satellite
	instrument5 - instrument
	infrared0 - mode
	image1 - mode
	star5 - direction
	star0 - direction
	star1 - direction
	star4 - direction
	groundstation3 - direction
	star2 - direction
	planet6 - direction
	phenomenon7 - direction
	star8 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation3)
	(calibration_target instrument0 star5)
	(supports instrument1 image1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star0)
	(calibration_target instrument1 star4)
	(supports instrument2 infrared0)
	(supports instrument2 image1)
	(calibration_target instrument2 star1)
	(supports instrument3 image1)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star4)
	(supports instrument4 image1)
	(supports instrument4 infrared0)
	(calibration_target instrument4 star4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star8)
	(supports instrument5 infrared0)
	(supports instrument5 image1)
	(calibration_target instrument5 star2)
	(calibration_target instrument5 groundstation3)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon7)
)
(:goal (and
	(pointing satellite0 star5)
	(have_image planet6 image1)
	(have_image phenomenon7 infrared0)
	(have_image star8 infrared0)
))

)
