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
	infrared2 - mode
	infrared1 - mode
	spectrograph0 - mode
	thermograph3 - mode
	groundstation3 - direction
	star4 - direction
	groundstation0 - direction
	groundstation1 - direction
	star2 - direction
	planet5 - direction
	star6 - direction
	planet7 - direction
	planet8 - direction
	planet9 - direction
	phenomenon10 - direction
	phenomenon11 - direction
	phenomenon12 - direction
)
(:init
	(supports instrument0 thermograph3)
	(supports instrument0 spectrograph0)
	(supports instrument0 infrared2)
	(calibration_target instrument0 groundstation1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument1 spectrograph0)
	(supports instrument1 infrared1)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 spectrograph0)
	(supports instrument2 thermograph3)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 infrared1)
	(calibration_target instrument4 star2)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet9)
)
(:goal (and
	(have_image planet5 thermograph3)
	(have_image star6 spectrograph0)
	(have_image planet7 thermograph3)
	(have_image planet8 infrared1)
	(have_image planet9 infrared2)
	(have_image phenomenon10 infrared1)
	(have_image phenomenon11 infrared1)
	(have_image phenomenon12 thermograph3)
))

)
