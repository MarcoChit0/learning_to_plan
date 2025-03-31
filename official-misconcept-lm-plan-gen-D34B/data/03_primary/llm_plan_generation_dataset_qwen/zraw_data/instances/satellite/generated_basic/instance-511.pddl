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
	thermograph3 - mode
	image2 - mode
	image0 - mode
	spectrograph1 - mode
	groundstation0 - direction
	star1 - direction
	star3 - direction
	star2 - direction
	groundstation4 - direction
	planet5 - direction
	star6 - direction
	planet7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 star2)
	(supports instrument1 thermograph3)
	(supports instrument1 image2)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 star2)
	(supports instrument2 spectrograph1)
	(supports instrument2 image2)
	(supports instrument2 thermograph3)
	(calibration_target instrument2 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon8)
	(supports instrument3 spectrograph1)
	(supports instrument3 image0)
	(calibration_target instrument3 star2)
	(supports instrument4 image0)
	(supports instrument4 image2)
	(calibration_target instrument4 groundstation4)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
)
(:goal (and
	(pointing satellite1 star6)
	(have_image planet5 spectrograph1)
	(have_image star6 thermograph3)
	(have_image planet7 image2)
	(have_image phenomenon8 thermograph3)
))

)
