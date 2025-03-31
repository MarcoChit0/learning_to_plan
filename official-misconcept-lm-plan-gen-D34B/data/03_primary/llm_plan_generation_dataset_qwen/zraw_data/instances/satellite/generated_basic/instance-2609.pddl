(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	infrared2 - mode
	infrared3 - mode
	image0 - mode
	image1 - mode
	groundstation1 - direction
	groundstation0 - direction
	phenomenon2 - direction
)
(:init
	(supports instrument0 infrared2)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 infrared2)
	(supports instrument1 image1)
	(calibration_target instrument1 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument2 image1)
	(supports instrument2 infrared2)
	(supports instrument2 infrared3)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 infrared2)
	(supports instrument3 image1)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 image1)
	(supports instrument4 infrared2)
	(supports instrument4 infrared3)
	(calibration_target instrument4 groundstation0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon2)
)
(:goal (and
	(pointing satellite0 groundstation0)
	(pointing satellite1 groundstation1)
	(have_image phenomenon2 image1)
))

)
