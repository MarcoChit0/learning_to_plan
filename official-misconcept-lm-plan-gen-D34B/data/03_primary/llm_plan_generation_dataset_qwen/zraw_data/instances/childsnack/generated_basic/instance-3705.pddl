; child-snack task with 2 children and 0.8 gluten factor 
; constant factor of 1.7
; random seed: 650403

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child66 child30 - child
    bread239 bread47 - bread-portion
    content303 content293 - content-portion
    tray41 tray156 - tray
    table363 table263 table245 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray41 kitchen)
     (at tray156 kitchen)
     (at_kitchen_bread bread239)
     (at_kitchen_bread bread47)
     (at_kitchen_content content303)
     (at_kitchen_content content293)
     (no_gluten_bread bread47)
     (no_gluten_content content303)
     (allergic_gluten child30)
     (not_allergic_gluten child66)
     (waiting child66 table363)
     (waiting child30 table363)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child66)
     (served child30)
    )
  )
)
