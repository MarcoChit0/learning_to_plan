; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.9
; random seed: 411909

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child85 child429 - child
    bread223 bread111 - bread-portion
    content111 content325 - content-portion
    tray335 tray461 tray432 tray237 - tray
    table320 table478 table187 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray335 kitchen)
     (at tray461 kitchen)
     (at tray432 kitchen)
     (at tray237 kitchen)
     (at_kitchen_bread bread223)
     (at_kitchen_bread bread111)
     (at_kitchen_content content111)
     (at_kitchen_content content325)
     (no_gluten_bread bread111)
     (no_gluten_content content325)
     (allergic_gluten child85)
     (not_allergic_gluten child429)
     (waiting child85 table320)
     (waiting child429 table187)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child85)
     (served child429)
    )
  )
)
