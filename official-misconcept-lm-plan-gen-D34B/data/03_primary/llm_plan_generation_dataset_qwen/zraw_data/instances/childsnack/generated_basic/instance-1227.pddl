; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 851125

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child4 child399 child330 - child
    bread352 bread34 bread99 - bread-portion
    content141 content218 content49 - content-portion
    tray432 tray121 tray485 - tray
    table478 table302 table366 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray432 kitchen)
     (at tray121 kitchen)
     (at tray485 kitchen)
     (at_kitchen_bread bread352)
     (at_kitchen_bread bread34)
     (at_kitchen_bread bread99)
     (at_kitchen_content content141)
     (at_kitchen_content content218)
     (at_kitchen_content content49)
     (not_allergic_gluten child4)
     (not_allergic_gluten child399)
     (not_allergic_gluten child330)
     (waiting child4 table478)
     (waiting child399 table478)
     (waiting child330 table478)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child4)
     (served child399)
     (served child330)
    )
  )
)
