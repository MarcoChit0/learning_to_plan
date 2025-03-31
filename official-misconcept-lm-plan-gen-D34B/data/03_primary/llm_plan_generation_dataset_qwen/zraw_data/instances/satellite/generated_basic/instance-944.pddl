(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	satellite2 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	spectrograph3 - mode
	spectrograph2 - mode
	image0 - mode
	image1 - mode
	groundstation0 - direction
	groundstation1 - direction
	groundstation3 - direction
	star2 - direction
	star4 - direction
	star5 - direction
	planet6 - direction
)
(:init
	(supports instrument0 spectrograph3)
	(supports instrument0 spectrograph2)
	(supports instrument0 image1)
	(calibration_target instrument0 groundstation3)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation3)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation3)
	(supports instrument3 spectrograph2)
	(supports instrument3 image0)
	(supports instrument3 image1)
	(calibration_target instrument3 groundstation3)
	(supports instrument4 image0)
	(supports instrument4 image1)
	(calibration_target instrument4 star2)
	(on_board instrument2 satellite2)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation3)
)
(:goal (and
	(have_image star4 image0)
	(have_image star5 image1)
	(have_image planet6 image0)
))

)
