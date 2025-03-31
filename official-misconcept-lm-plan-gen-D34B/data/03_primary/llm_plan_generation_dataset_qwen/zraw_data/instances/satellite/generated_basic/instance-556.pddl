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
	thermograph1 - mode
	spectrograph0 - mode
	image2 - mode
	groundstation0 - direction
	groundstation1 - direction
	star2 - direction
	star3 - direction
	planet4 - direction
	star5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(supports instrument0 thermograph1)
	(supports instrument0 image2)
	(calibration_target instrument0 groundstation1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument1 spectrograph0)
	(supports instrument1 image2)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 groundstation1)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star5)
	(supports instrument2 spectrograph0)
	(supports instrument2 image2)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 thermograph1)
	(supports instrument3 image2)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 thermograph1)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 groundstation1)
	(on_board instrument2 satellite2)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
)
(:goal (and
	(have_image star2 thermograph1)
	(have_image star3 image2)
	(have_image planet4 thermograph1)
	(have_image star5 image2)
	(have_image phenomenon6 thermograph1)
))

)
