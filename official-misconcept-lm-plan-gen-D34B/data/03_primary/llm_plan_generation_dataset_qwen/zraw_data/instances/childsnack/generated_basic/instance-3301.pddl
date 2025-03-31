; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 847793

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child432 child424 - child
    bread170 bread416 - bread-portion
    content153 content413 - content-portion
    tray25 tray446 tray241 tray9 - tray
    table317 table120 table110 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray25 kitchen)
     (at tray446 kitchen)
     (at tray241 kitchen)
     (at tray9 kitchen)
     (at_kitchen_bread bread170)
     (at_kitchen_bread bread416)
     (at_kitchen_content content153)
     (at_kitchen_content content413)
     (no_gluten_bread bread416)
     (no_gluten_content content413)
     (allergic_gluten child424)
     (not_allergic_gluten child432)
     (waiting child432 table110)
     (waiting child424 table120)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child432)
     (served child424)
    )
  )
)
