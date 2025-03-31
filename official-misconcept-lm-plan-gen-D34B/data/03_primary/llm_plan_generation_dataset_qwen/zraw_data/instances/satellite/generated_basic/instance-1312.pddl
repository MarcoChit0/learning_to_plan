(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite1 - satellite
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	thermograph2 - mode
	infrared1 - mode
	spectrograph0 - mode
	star0 - direction
	groundstation2 - direction
	groundstation1 - direction
	star5 - direction
	star3 - direction
	groundstation4 - direction
	star6 - direction
	phenomenon7 - direction
	star8 - direction
	planet9 - direction
)
(:init
	(supports instrument0 thermograph2)
	(supports instrument0 infrared1)
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 star3)
	(calibration_target instrument0 star5)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 star0)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 infrared1)
	(supports instrument2 thermograph2)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 groundstation4)
	(calibration_target instrument2 star5)
	(supports instrument3 infrared1)
	(supports instrument3 spectrograph0)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 thermograph2)
	(calibration_target instrument4 groundstation1)
	(calibration_target instrument4 groundstation4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon7)
	(supports instrument5 thermograph2)
	(supports instrument5 infrared1)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 star5)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument6 thermograph2)
	(supports instrument6 infrared1)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 groundstation4)
	(calibration_target instrument6 star3)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
)
(:goal (and
	(pointing satellite0 phenomenon7)
	(pointing satellite1 star8)
	(have_image star6 thermograph2)
	(have_image phenomenon7 infrared1)
	(have_image star8 infrared1)
	(have_image planet9 spectrograph0)
))

)
