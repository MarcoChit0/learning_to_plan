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
	satellite2 - satellite
	instrument5 - instrument
	image0 - mode
	image1 - mode
	infrared2 - mode
	spectrograph3 - mode
	groundstation5 - direction
	star2 - direction
	star3 - direction
	star4 - direction
	star1 - direction
	groundstation0 - direction
	star6 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 spectrograph3)
	(supports instrument0 infrared2)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 image0)
	(supports instrument1 spectrograph3)
	(calibration_target instrument1 star2)
	(calibration_target instrument1 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument2 image1)
	(supports instrument2 infrared2)
	(calibration_target instrument2 star3)
	(supports instrument3 spectrograph3)
	(supports instrument3 infrared2)
	(supports instrument3 image1)
	(calibration_target instrument3 star4)
	(supports instrument4 image0)
	(supports instrument4 spectrograph3)
	(calibration_target instrument4 star4)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
	(supports instrument5 infrared2)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation0)
	(calibration_target instrument5 star1)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
)
(:goal (and
	(have_image star6 infrared2)
))

)
