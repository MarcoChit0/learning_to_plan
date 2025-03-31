; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.3
; random seed: 393320

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child136 child226 - child
    bread296 bread107 - bread-portion
    content300 content130 - content-portion
    tray130 - tray
    table63 table228 table394 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray130 kitchen)
     (at_kitchen_bread bread296)
     (at_kitchen_bread bread107)
     (at_kitchen_content content300)
     (at_kitchen_content content130)
     (no_gluten_bread bread107)
     (no_gluten_content content300)
     (allergic_gluten child136)
     (not_allergic_gluten child226)
     (waiting child136 table63)
     (waiting child226 table394)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child136)
     (served child226)
    )
  )
)
