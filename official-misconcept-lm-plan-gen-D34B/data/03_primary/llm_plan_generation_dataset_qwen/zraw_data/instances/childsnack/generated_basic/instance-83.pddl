; child-snack task with 2 children and 0.36 gluten factor 
; constant factor of 1.9
; random seed: 748823

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child144 child102 - child
    bread309 bread445 - bread-portion
    content51 content261 - content-portion
    tray0 tray34 - tray
    table255 table182 table422 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray0 kitchen)
     (at tray34 kitchen)
     (at_kitchen_bread bread309)
     (at_kitchen_bread bread445)
     (at_kitchen_content content51)
     (at_kitchen_content content261)
     (not_allergic_gluten child102)
     (not_allergic_gluten child144)
     (waiting child144 table182)
     (waiting child102 table255)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child144)
     (served child102)
    )
  )
)
