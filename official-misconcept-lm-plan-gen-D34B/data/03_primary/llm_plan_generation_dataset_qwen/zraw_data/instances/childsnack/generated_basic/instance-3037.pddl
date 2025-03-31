; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 710764

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child215 child120 - child
    bread160 bread48 - bread-portion
    content277 content99 - content-portion
    tray62 tray226 tray419 - tray
    table281 table19 table457 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray62 kitchen)
     (at tray226 kitchen)
     (at tray419 kitchen)
     (at_kitchen_bread bread160)
     (at_kitchen_bread bread48)
     (at_kitchen_content content277)
     (at_kitchen_content content99)
     (no_gluten_bread bread160)
     (no_gluten_content content277)
     (allergic_gluten child120)
     (not_allergic_gluten child215)
     (waiting child215 table281)
     (waiting child120 table457)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child215)
     (served child120)
    )
  )
)
