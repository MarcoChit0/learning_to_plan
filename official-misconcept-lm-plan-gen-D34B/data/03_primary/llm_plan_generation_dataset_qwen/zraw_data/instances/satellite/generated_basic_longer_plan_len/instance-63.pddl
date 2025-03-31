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
	satellite3 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	satellite4 - satellite
	instrument8 - instrument
	thermograph1 - mode
	infrared2 - mode
	spectrograph0 - mode
	image3 - mode
	groundstation1 - direction
	star0 - direction
	groundstation3 - direction
	groundstation2 - direction
	phenomenon4 - direction
	planet5 - direction
	planet6 - direction
	phenomenon7 - direction
	star8 - direction
	planet9 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(supports instrument0 image3)
	(supports instrument0 infrared2)
	(calibration_target instrument0 groundstation2)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon7)
	(supports instrument1 spectrograph0)
	(supports instrument1 image3)
	(supports instrument1 infrared2)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 image3)
	(supports instrument2 spectrograph0)
	(supports instrument2 infrared2)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 groundstation2)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star8)
	(supports instrument4 spectrograph0)
	(supports instrument4 image3)
	(calibration_target instrument4 groundstation2)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet6)
	(supports instrument5 spectrograph0)
	(supports instrument5 thermograph1)
	(supports instrument5 infrared2)
	(calibration_target instrument5 star0)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 groundstation2)
	(supports instrument7 spectrograph0)
	(supports instrument7 infrared2)
	(calibration_target instrument7 groundstation3)
	(on_board instrument5 satellite3)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation3)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 groundstation2)
	(on_board instrument8 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star8)
)
(:goal (and
	(pointing satellite1 groundstation2)
	(pointing satellite2 groundstation1)
	(pointing satellite3 groundstation1)
	(have_image phenomenon4 thermograph1)
	(have_image planet5 thermograph1)
	(have_image planet6 spectrograph0)
	(have_image phenomenon7 spectrograph0)
	(have_image star8 spectrograph0)
	(have_image planet9 infrared2)
))

)
