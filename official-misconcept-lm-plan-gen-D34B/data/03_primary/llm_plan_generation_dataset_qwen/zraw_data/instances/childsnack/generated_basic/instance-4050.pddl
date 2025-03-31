; child-snack task with 3 children and 0.22 gluten factor 
; constant factor of 1.9
; random seed: 261703

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child309 child132 child235 - child
    bread328 bread100 bread94 - bread-portion
    content389 content105 content163 - content-portion
    tray1 - tray
    table165 table198 table298 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray1 kitchen)
     (at_kitchen_bread bread328)
     (at_kitchen_bread bread100)
     (at_kitchen_bread bread94)
     (at_kitchen_content content389)
     (at_kitchen_content content105)
     (at_kitchen_content content163)
     (not_allergic_gluten child309)
     (not_allergic_gluten child132)
     (not_allergic_gluten child235)
     (waiting child309 table298)
     (waiting child132 table198)
     (waiting child235 table298)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child309)
     (served child132)
     (served child235)
    )
  )
)
