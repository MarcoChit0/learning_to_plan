; child-snack task with 2 children and 0.48 gluten factor 
; constant factor of 1.9
; random seed: 932103

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child224 child387 - child
    bread110 bread20 - bread-portion
    content72 content445 - content-portion
    tray398 tray83 - tray
    table350 table24 table132 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray398 kitchen)
     (at tray83 kitchen)
     (at_kitchen_bread bread110)
     (at_kitchen_bread bread20)
     (at_kitchen_content content72)
     (at_kitchen_content content445)
     (not_allergic_gluten child387)
     (not_allergic_gluten child224)
     (waiting child224 table24)
     (waiting child387 table132)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child224)
     (served child387)
    )
  )
)
