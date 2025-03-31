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
	thermograph3 - mode
	image2 - mode
	spectrograph1 - mode
	thermograph0 - mode
	groundstation0 - direction
	star2 - direction
	groundstation3 - direction
	groundstation1 - direction
	groundstation4 - direction
	star5 - direction
	star6 - direction
	star7 - direction
	planet8 - direction
	star9 - direction
	phenomenon10 - direction
	star11 - direction
	planet12 - direction
)
(:init
	(supports instrument0 thermograph3)
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation3)
	(supports instrument2 thermograph0)
	(supports instrument2 image2)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 groundstation3)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 thermograph3)
	(supports instrument4 spectrograph1)
	(supports instrument4 image2)
	(calibration_target instrument4 groundstation4)
	(supports instrument5 spectrograph1)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 groundstation4)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet12)
)
(:goal (and
	(pointing satellite1 star9)
	(have_image star5 thermograph0)
	(have_image star6 spectrograph1)
	(have_image star7 thermograph3)
	(have_image planet8 image2)
	(have_image star9 thermograph0)
	(have_image phenomenon10 image2)
	(have_image star11 thermograph3)
	(have_image planet12 thermograph3)
))

)
