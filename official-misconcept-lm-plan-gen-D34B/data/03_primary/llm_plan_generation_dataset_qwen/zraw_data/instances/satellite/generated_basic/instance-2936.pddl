(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	image0 - mode
	image1 - mode
	spectrograph2 - mode
	star1 - direction
	star3 - direction
	star4 - direction
	groundstation2 - direction
	groundstation0 - direction
	star5 - direction
	planet6 - direction
	phenomenon7 - direction
	star8 - direction
)
(:init
	(supports instrument0 spectrograph2)
	(supports instrument0 image1)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 image1)
	(supports instrument1 image0)
	(calibration_target instrument1 star4)
	(supports instrument2 image1)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument3 image0)
	(supports instrument3 image1)
	(supports instrument3 spectrograph2)
	(calibration_target instrument3 groundstation0)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
)
(:goal (and
	(pointing satellite0 groundstation2)
	(have_image star5 spectrograph2)
	(have_image planet6 image0)
	(have_image phenomenon7 image0)
	(have_image star8 image0)
))

)
