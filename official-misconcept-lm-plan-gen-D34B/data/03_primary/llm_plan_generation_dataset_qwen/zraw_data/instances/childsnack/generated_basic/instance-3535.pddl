; child-snack task with 2 children and 0.42 gluten factor 
; constant factor of 1.3
; random seed: 690223

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child445 child108 - child
    bread132 bread283 - bread-portion
    content114 content39 - content-portion
    tray467 tray134 - tray
    table95 table362 table93 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray467 kitchen)
     (at tray134 kitchen)
     (at_kitchen_bread bread132)
     (at_kitchen_bread bread283)
     (at_kitchen_content content114)
     (at_kitchen_content content39)
     (not_allergic_gluten child445)
     (not_allergic_gluten child108)
     (waiting child445 table362)
     (waiting child108 table362)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child445)
     (served child108)
    )
  )
)
