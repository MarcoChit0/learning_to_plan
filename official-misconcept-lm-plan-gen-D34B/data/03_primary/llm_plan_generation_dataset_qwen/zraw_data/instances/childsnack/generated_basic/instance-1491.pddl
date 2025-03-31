; child-snack task with 2 children and 0.66 gluten factor 
; constant factor of 1.3
; random seed: 255586

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child238 child410 - child
    bread417 bread499 - bread-portion
    content332 content155 - content-portion
    tray105 tray418 tray385 - tray
    table463 table461 table346 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray105 kitchen)
     (at tray418 kitchen)
     (at tray385 kitchen)
     (at_kitchen_bread bread417)
     (at_kitchen_bread bread499)
     (at_kitchen_content content332)
     (at_kitchen_content content155)
     (no_gluten_bread bread499)
     (no_gluten_content content155)
     (allergic_gluten child410)
     (not_allergic_gluten child238)
     (waiting child238 table463)
     (waiting child410 table461)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child238)
     (served child410)
    )
  )
)
