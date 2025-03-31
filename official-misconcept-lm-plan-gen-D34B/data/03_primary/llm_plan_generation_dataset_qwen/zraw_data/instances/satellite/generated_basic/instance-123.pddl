(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	spectrograph1 - mode
	thermograph0 - mode
	infrared2 - mode
	groundstation4 - direction
	star2 - direction
	star0 - direction
	star3 - direction
	groundstation5 - direction
	groundstation1 - direction
	planet6 - direction
	star7 - direction
	planet8 - direction
	phenomenon9 - direction
	planet10 - direction
	star11 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(supports instrument0 thermograph0)
	(supports instrument0 infrared2)
	(calibration_target instrument0 star2)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 star0)
	(supports instrument2 infrared2)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star3)
	(calibration_target instrument2 star0)
	(supports instrument3 infrared2)
	(supports instrument3 spectrograph1)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 groundstation5)
	(calibration_target instrument3 groundstation1)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
	(supports instrument4 spectrograph1)
	(calibration_target instrument4 groundstation1)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
)
(:goal (and
	(pointing satellite1 star3)
	(pointing satellite2 groundstation5)
	(have_image planet6 thermograph0)
	(have_image star7 thermograph0)
	(have_image planet8 infrared2)
	(have_image phenomenon9 infrared2)
	(have_image planet10 infrared2)
	(have_image star11 spectrograph1)
))

)
