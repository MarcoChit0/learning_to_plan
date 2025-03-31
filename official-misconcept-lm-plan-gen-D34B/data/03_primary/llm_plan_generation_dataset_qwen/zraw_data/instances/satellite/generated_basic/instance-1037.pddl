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
	infrared1 - mode
	spectrograph0 - mode
	thermograph2 - mode
	star0 - direction
	groundstation1 - direction
	star3 - direction
	groundstation2 - direction
	planet4 - direction
	star5 - direction
	phenomenon6 - direction
	star7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(supports instrument0 infrared1)
	(calibration_target instrument0 groundstation2)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet4)
	(supports instrument1 thermograph2)
	(supports instrument1 infrared1)
	(calibration_target instrument1 star3)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star5)
	(supports instrument2 spectrograph0)
	(supports instrument2 infrared1)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 star3)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 star3)
	(supports instrument4 spectrograph0)
	(supports instrument4 thermograph2)
	(supports instrument4 infrared1)
	(calibration_target instrument4 groundstation2)
	(on_board instrument2 satellite2)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet4)
)
(:goal (and
	(pointing satellite0 planet4)
	(have_image planet4 spectrograph0)
	(have_image star5 infrared1)
	(have_image phenomenon6 infrared1)
	(have_image star7 thermograph2)
	(have_image planet8 infrared1)
))

)
