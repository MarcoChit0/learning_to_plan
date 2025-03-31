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
	spectrograph3 - mode
	thermograph0 - mode
	image2 - mode
	image1 - mode
	star4 - direction
	groundstation2 - direction
	groundstation3 - direction
	star0 - direction
	groundstation1 - direction
	phenomenon5 - direction
	star6 - direction
	phenomenon7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 thermograph0)
	(supports instrument1 image1)
	(calibration_target instrument1 star0)
	(supports instrument2 image1)
	(calibration_target instrument2 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon5)
	(supports instrument3 image1)
	(calibration_target instrument3 star0)
	(supports instrument4 image2)
	(supports instrument4 image1)
	(supports instrument4 spectrograph3)
	(calibration_target instrument4 groundstation1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon8)
)
(:goal (and
	(pointing satellite0 phenomenon8)
	(have_image phenomenon5 image2)
	(have_image star6 thermograph0)
	(have_image phenomenon7 thermograph0)
	(have_image phenomenon8 image1)
))

)
