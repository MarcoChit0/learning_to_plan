(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	infrared0 - mode
	image2 - mode
	image3 - mode
	infrared1 - mode
	groundstation1 - direction
	star2 - direction
	star3 - direction
	groundstation0 - direction
	star5 - direction
	groundstation4 - direction
	planet6 - direction
)
(:init
	(supports instrument0 image3)
	(supports instrument0 image2)
	(calibration_target instrument0 groundstation4)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument1 infrared0)
	(supports instrument1 image2)
	(supports instrument1 image3)
	(calibration_target instrument1 star3)
	(supports instrument2 image2)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 image2)
	(supports instrument3 infrared1)
	(supports instrument3 image3)
	(calibration_target instrument3 star5)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
	(supports instrument4 infrared1)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation4)
	(calibration_target instrument4 star5)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
)
(:goal (and
	(pointing satellite1 star5)
	(have_image planet6 infrared0)
))

)
