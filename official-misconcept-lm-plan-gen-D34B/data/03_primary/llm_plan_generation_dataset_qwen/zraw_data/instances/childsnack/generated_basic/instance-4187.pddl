; child-snack task with 2 children and 0.28 gluten factor 
; constant factor of 1.7
; random seed: 155947

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child313 child213 - child
    bread445 bread224 - bread-portion
    content305 content382 - content-portion
    tray353 - tray
    table320 table63 table497 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray353 kitchen)
     (at_kitchen_bread bread445)
     (at_kitchen_bread bread224)
     (at_kitchen_content content305)
     (at_kitchen_content content382)
     (not_allergic_gluten child313)
     (not_allergic_gluten child213)
     (waiting child313 table497)
     (waiting child213 table320)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child313)
     (served child213)
    )
  )
)
