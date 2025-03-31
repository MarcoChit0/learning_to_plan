(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite1 - satellite
	instrument4 - instrument
	instrument5 - instrument
	image2 - mode
	spectrograph1 - mode
	infrared0 - mode
	groundstation0 - direction
	star1 - direction
	star2 - direction
)
(:init
	(supports instrument0 image2)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 image2)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 image2)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 image2)
	(calibration_target instrument3 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 image2)
	(calibration_target instrument5 groundstation0)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
)
(:goal (and
	(pointing satellite1 groundstation0)
	(have_image star1 infrared0)
	(have_image star2 infrared0)
))

)
