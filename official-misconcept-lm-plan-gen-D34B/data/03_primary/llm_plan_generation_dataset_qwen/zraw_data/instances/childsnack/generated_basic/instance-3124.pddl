; child-snack task with 2 children and 0.65 gluten factor 
; constant factor of 1.3
; random seed: 583441

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child444 child335 - child
    bread419 bread151 - bread-portion
    content224 content111 - content-portion
    tray252 tray135 tray346 tray19 - tray
    table204 table229 table459 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray252 kitchen)
     (at tray135 kitchen)
     (at tray346 kitchen)
     (at tray19 kitchen)
     (at_kitchen_bread bread419)
     (at_kitchen_bread bread151)
     (at_kitchen_content content224)
     (at_kitchen_content content111)
     (no_gluten_bread bread419)
     (no_gluten_content content111)
     (allergic_gluten child335)
     (not_allergic_gluten child444)
     (waiting child444 table229)
     (waiting child335 table459)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child444)
     (served child335)
    )
  )
)
