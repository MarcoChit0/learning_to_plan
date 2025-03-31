; child-snack task with 2 children and 0.59 gluten factor 
; constant factor of 1.9
; random seed: 388929

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child431 child320 - child
    bread304 bread85 - bread-portion
    content319 content240 - content-portion
    tray60 tray108 - tray
    table445 table393 table295 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray60 kitchen)
     (at tray108 kitchen)
     (at_kitchen_bread bread304)
     (at_kitchen_bread bread85)
     (at_kitchen_content content319)
     (at_kitchen_content content240)
     (no_gluten_bread bread304)
     (no_gluten_content content319)
     (allergic_gluten child431)
     (not_allergic_gluten child320)
     (waiting child431 table445)
     (waiting child320 table295)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child431)
     (served child320)
    )
  )
)
