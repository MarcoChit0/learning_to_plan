; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 114701

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child477 child253 - child
    bread104 bread197 - bread-portion
    content418 content120 - content-portion
    tray239 tray482 tray375 tray492 - tray
    table370 table171 table61 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray239 kitchen)
     (at tray482 kitchen)
     (at tray375 kitchen)
     (at tray492 kitchen)
     (at_kitchen_bread bread104)
     (at_kitchen_bread bread197)
     (at_kitchen_content content418)
     (at_kitchen_content content120)
     (no_gluten_bread bread104)
     (no_gluten_content content418)
     (allergic_gluten child253)
     (not_allergic_gluten child477)
     (waiting child477 table61)
     (waiting child253 table171)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child477)
     (served child253)
    )
  )
)
