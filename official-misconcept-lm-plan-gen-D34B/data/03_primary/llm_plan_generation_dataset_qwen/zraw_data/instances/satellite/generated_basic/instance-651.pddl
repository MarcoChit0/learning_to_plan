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
	spectrograph0 - mode
	image1 - mode
	star4 - direction
	star2 - direction
	star5 - direction
	star3 - direction
	groundstation1 - direction
	groundstation0 - direction
	planet6 - direction
	planet7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 image1)
	(calibration_target instrument0 star4)
	(calibration_target instrument0 star5)
	(supports instrument1 image1)
	(calibration_target instrument1 star2)
	(supports instrument2 image1)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 star5)
	(calibration_target instrument2 star3)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 groundstation1)
	(calibration_target instrument3 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 groundstation0)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star4)
)
(:goal (and
	(have_image planet6 image1)
	(have_image planet7 image1)
	(have_image phenomenon8 image1)
))

)
