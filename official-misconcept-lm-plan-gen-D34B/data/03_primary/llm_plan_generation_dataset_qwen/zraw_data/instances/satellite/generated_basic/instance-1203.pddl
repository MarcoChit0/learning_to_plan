(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	spectrograph0 - mode
	image2 - mode
	spectrograph3 - mode
	thermograph1 - mode
	groundstation2 - direction
	star1 - direction
	groundstation0 - direction
	star3 - direction
	star4 - direction
	phenomenon5 - direction
)
(:init
	(supports instrument0 image2)
	(supports instrument0 spectrograph3)
	(calibration_target instrument0 star1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon5)
	(supports instrument1 spectrograph3)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 star1)
	(supports instrument2 spectrograph3)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 star1)
	(supports instrument3 image2)
	(supports instrument3 spectrograph0)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 spectrograph0)
	(supports instrument4 image2)
	(calibration_target instrument4 groundstation0)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
)
(:goal (and
	(pointing satellite0 phenomenon5)
	(pointing satellite1 groundstation2)
	(have_image star3 thermograph1)
	(have_image star4 spectrograph0)
	(have_image phenomenon5 spectrograph0)
))

)
