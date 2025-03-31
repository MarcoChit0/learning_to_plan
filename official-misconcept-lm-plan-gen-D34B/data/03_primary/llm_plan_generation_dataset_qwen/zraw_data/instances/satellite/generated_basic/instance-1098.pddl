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
	infrared2 - mode
	image1 - mode
	infrared0 - mode
	star1 - direction
	groundstation2 - direction
	star3 - direction
	star4 - direction
	star0 - direction
	star5 - direction
	phenomenon6 - direction
	phenomenon7 - direction
	star8 - direction
)
(:init
	(supports instrument0 infrared2)
	(supports instrument0 image1)
	(calibration_target instrument0 star3)
	(calibration_target instrument0 star1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation2)
	(calibration_target instrument1 star5)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument2 image1)
	(calibration_target instrument2 star3)
	(supports instrument3 infrared2)
	(supports instrument3 image1)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star4)
	(calibration_target instrument3 star5)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument4 image1)
	(supports instrument4 infrared0)
	(supports instrument4 infrared2)
	(calibration_target instrument4 star5)
	(calibration_target instrument4 star0)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon6)
)
(:goal (and
	(pointing satellite2 star5)
	(have_image phenomenon6 infrared0)
	(have_image phenomenon7 infrared0)
	(have_image star8 infrared0)
))

)
