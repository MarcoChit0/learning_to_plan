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
	instrument5 - instrument
	spectrograph1 - mode
	infrared0 - mode
	infrared2 - mode
	groundstation5 - direction
	star4 - direction
	groundstation3 - direction
	star2 - direction
	star0 - direction
	groundstation1 - direction
	planet6 - direction
	star7 - direction
	planet8 - direction
	planet9 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(supports instrument0 infrared2)
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation3)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument1 infrared2)
	(supports instrument1 infrared0)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 star4)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation5)
	(supports instrument2 spectrograph1)
	(supports instrument2 infrared2)
	(calibration_target instrument2 groundstation3)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation3)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 infrared2)
	(calibration_target instrument4 star2)
	(calibration_target instrument4 star0)
	(supports instrument5 infrared0)
	(supports instrument5 spectrograph1)
	(supports instrument5 infrared2)
	(calibration_target instrument5 groundstation1)
	(calibration_target instrument5 star0)
	(on_board instrument2 satellite2)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
)
(:goal (and
	(pointing satellite0 planet6)
	(pointing satellite2 star7)
	(have_image planet6 infrared2)
	(have_image star7 spectrograph1)
	(have_image planet8 spectrograph1)
	(have_image planet9 infrared2)
))

)
