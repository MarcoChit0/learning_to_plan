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
	instrument5 - instrument
	image1 - mode
	spectrograph3 - mode
	image2 - mode
	image0 - mode
	groundstation3 - direction
	groundstation2 - direction
	star1 - direction
	groundstation0 - direction
	planet4 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 image0)
	(calibration_target instrument0 star1)
	(supports instrument1 spectrograph3)
	(calibration_target instrument1 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation3)
	(supports instrument2 image0)
	(supports instrument2 spectrograph3)
	(calibration_target instrument2 groundstation3)
	(supports instrument3 spectrograph3)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 spectrograph3)
	(supports instrument4 image2)
	(supports instrument4 image0)
	(calibration_target instrument4 star1)
	(supports instrument5 spectrograph3)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
)
(:goal (and
	(pointing satellite1 groundstation3)
	(have_image planet4 spectrograph3)
))

)
