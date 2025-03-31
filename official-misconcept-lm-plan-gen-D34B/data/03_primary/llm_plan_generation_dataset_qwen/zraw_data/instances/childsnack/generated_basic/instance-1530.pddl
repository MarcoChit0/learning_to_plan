; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 51553

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child270 child227 - child
    bread498 bread65 - bread-portion
    content64 content466 - content-portion
    tray146 tray342 tray17 tray290 - tray
    table257 table295 table53 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray146 kitchen)
     (at tray342 kitchen)
     (at tray17 kitchen)
     (at tray290 kitchen)
     (at_kitchen_bread bread498)
     (at_kitchen_bread bread65)
     (at_kitchen_content content64)
     (at_kitchen_content content466)
     (not_allergic_gluten child270)
     (not_allergic_gluten child227)
     (waiting child270 table257)
     (waiting child227 table257)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child270)
     (served child227)
    )
  )
)
