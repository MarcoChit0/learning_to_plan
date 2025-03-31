(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	image2 - mode
	spectrograph0 - mode
	image1 - mode
	image3 - mode
	star2 - direction
	star1 - direction
	groundstation3 - direction
	groundstation0 - direction
	star4 - direction
	star5 - direction
	planet6 - direction
	star7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 image2)
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 image2)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument3 spectrograph0)
	(supports instrument3 image3)
	(calibration_target instrument3 groundstation3)
	(supports instrument4 image2)
	(supports instrument4 image1)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 spectrograph0)
	(supports instrument5 image3)
	(supports instrument5 image1)
	(calibration_target instrument5 star4)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star7)
)
(:goal (and
	(pointing satellite1 star7)
	(have_image star5 image2)
	(have_image planet6 image1)
	(have_image star7 image3)
	(have_image planet8 image1)
))

)
