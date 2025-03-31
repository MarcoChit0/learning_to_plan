; child-snack task with 2 children and 0.31 gluten factor 
; constant factor of 1.7
; random seed: 188687

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child385 child326 - child
    bread442 bread95 - bread-portion
    content4 content218 - content-portion
    tray21 tray434 tray126 - tray
    table1 table247 table477 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray21 kitchen)
     (at tray434 kitchen)
     (at tray126 kitchen)
     (at_kitchen_bread bread442)
     (at_kitchen_bread bread95)
     (at_kitchen_content content4)
     (at_kitchen_content content218)
     (not_allergic_gluten child385)
     (not_allergic_gluten child326)
     (waiting child385 table247)
     (waiting child326 table247)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child385)
     (served child326)
    )
  )
)
