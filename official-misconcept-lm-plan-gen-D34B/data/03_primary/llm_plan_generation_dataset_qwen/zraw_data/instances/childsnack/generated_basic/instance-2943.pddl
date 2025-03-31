; child-snack task with 1 children and 0.69 gluten factor 
; constant factor of 1.3
; random seed: 542997

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child371 - child
    bread132 - bread-portion
    content491 - content-portion
    tray472 tray225 tray19 tray70 - tray
    table18 table46 table125 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray472 kitchen)
     (at tray225 kitchen)
     (at tray19 kitchen)
     (at tray70 kitchen)
     (at_kitchen_bread bread132)
     (at_kitchen_content content491)
     (not_allergic_gluten child371)
     (waiting child371 table18)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child371)
    )
  )
)
