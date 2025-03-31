(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	image0 - mode
	infrared1 - mode
	star1 - direction
	star0 - direction
	groundstation2 - direction
	phenomenon3 - direction
	phenomenon4 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 infrared1)
	(calibration_target instrument0 star1)
	(supports instrument1 image0)
	(supports instrument1 infrared1)
	(calibration_target instrument1 star0)
	(supports instrument2 image0)
	(supports instrument2 infrared1)
	(calibration_target instrument2 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument3 image0)
	(supports instrument3 infrared1)
	(calibration_target instrument3 star0)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument4 infrared1)
	(supports instrument4 image0)
	(calibration_target instrument4 groundstation2)
	(supports instrument5 image0)
	(supports instrument5 infrared1)
	(calibration_target instrument5 groundstation2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon3)
)
(:goal (and
	(have_image phenomenon3 image0)
	(have_image phenomenon4 infrared1)
))

)
