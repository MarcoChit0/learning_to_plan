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
	satellite3 - satellite
	instrument4 - instrument
	satellite4 - satellite
	instrument5 - instrument
	image1 - mode
	infrared0 - mode
	star0 - direction
	star1 - direction
	phenomenon2 - direction
	star3 - direction
	planet4 - direction
)
(:init
	(supports instrument0 infrared0)
	(supports instrument0 image1)
	(calibration_target instrument0 star0)
	(supports instrument1 infrared0)
	(supports instrument1 image1)
	(calibration_target instrument1 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument2 infrared0)
	(supports instrument2 image1)
	(calibration_target instrument2 star0)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon2)
	(supports instrument3 image1)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star0)
	(on_board instrument3 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon2)
	(supports instrument4 infrared0)
	(calibration_target instrument4 star0)
	(on_board instrument4 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon2)
	(supports instrument5 infrared0)
	(supports instrument5 image1)
	(calibration_target instrument5 star0)
	(on_board instrument5 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star0)
)
(:goal (and
	(pointing satellite0 star1)
	(pointing satellite1 phenomenon2)
	(pointing satellite2 phenomenon2)
	(have_image star1 infrared0)
	(have_image phenomenon2 infrared0)
	(have_image star3 infrared0)
	(have_image planet4 image1)
))

)
