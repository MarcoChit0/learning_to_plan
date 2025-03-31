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
	instrument5 - instrument
	image1 - mode
	infrared0 - mode
	spectrograph2 - mode
	spectrograph3 - mode
	star4 - direction
	star2 - direction
	star1 - direction
	star3 - direction
	groundstation0 - direction
	groundstation5 - direction
	planet6 - direction
)
(:init
	(supports instrument0 infrared0)
	(supports instrument0 spectrograph3)
	(calibration_target instrument0 groundstation0)
	(calibration_target instrument0 star4)
	(supports instrument1 image1)
	(calibration_target instrument1 star3)
	(calibration_target instrument1 star4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument2 spectrograph2)
	(supports instrument2 image1)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star2)
	(calibration_target instrument2 star3)
	(supports instrument3 image1)
	(calibration_target instrument3 star1)
	(calibration_target instrument3 star3)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation5)
	(calibration_target instrument4 star3)
	(supports instrument5 spectrograph2)
	(calibration_target instrument5 groundstation5)
	(calibration_target instrument5 groundstation0)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet6)
)
(:goal (and
	(pointing satellite0 star3)
	(have_image planet6 spectrograph3)
))

)
