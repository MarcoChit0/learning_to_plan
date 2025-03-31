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
	instrument4 - instrument
	infrared0 - mode
	image2 - mode
	image1 - mode
	spectrograph3 - mode
	groundstation4 - direction
	groundstation3 - direction
	star2 - direction
	star1 - direction
	groundstation5 - direction
	groundstation0 - direction
	planet6 - direction
)
(:init
	(supports instrument0 image1)
	(calibration_target instrument0 groundstation3)
	(calibration_target instrument0 groundstation4)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument1 infrared0)
	(supports instrument1 image2)
	(calibration_target instrument1 groundstation5)
	(supports instrument2 image1)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation0)
	(calibration_target instrument2 star2)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation5)
	(supports instrument3 image2)
	(supports instrument3 image1)
	(calibration_target instrument3 star1)
	(supports instrument4 spectrograph3)
	(calibration_target instrument4 groundstation0)
	(calibration_target instrument4 groundstation5)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
)
(:goal (and
	(pointing satellite0 groundstation4)
	(pointing satellite1 groundstation0)
	(have_image planet6 infrared0)
))

)
