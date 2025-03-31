; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 474078

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child432 child113 - child
    bread10 bread277 - bread-portion
    content359 content250 - content-portion
    tray70 tray474 tray107 tray26 - tray
    table393 table333 table365 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray70 kitchen)
     (at tray474 kitchen)
     (at tray107 kitchen)
     (at tray26 kitchen)
     (at_kitchen_bread bread10)
     (at_kitchen_bread bread277)
     (at_kitchen_content content359)
     (at_kitchen_content content250)
     (no_gluten_bread bread10)
     (no_gluten_content content250)
     (allergic_gluten child113)
     (not_allergic_gluten child432)
     (waiting child432 table365)
     (waiting child113 table393)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child432)
     (served child113)
    )
  )
)
