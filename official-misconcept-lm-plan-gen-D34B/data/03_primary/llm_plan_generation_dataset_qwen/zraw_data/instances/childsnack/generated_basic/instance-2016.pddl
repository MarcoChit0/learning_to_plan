; child-snack task with 2 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 958192

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child390 child71 - child
    bread317 bread385 - bread-portion
    content207 content445 - content-portion
    tray272 tray15 tray432 - tray
    table249 table401 table456 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray272 kitchen)
     (at tray15 kitchen)
     (at tray432 kitchen)
     (at_kitchen_bread bread317)
     (at_kitchen_bread bread385)
     (at_kitchen_content content207)
     (at_kitchen_content content445)
     (not_allergic_gluten child390)
     (not_allergic_gluten child71)
     (waiting child390 table249)
     (waiting child71 table456)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child390)
     (served child71)
    )
  )
)
