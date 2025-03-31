; child-snack task with 3 children and 0.23 gluten factor 
; constant factor of 1.9
; random seed: 286181

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child474 child90 child233 - child
    bread137 bread319 bread382 - bread-portion
    content290 content325 content375 - content-portion
    tray334 tray102 tray79 - tray
    table63 table166 table125 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray334 kitchen)
     (at tray102 kitchen)
     (at tray79 kitchen)
     (at_kitchen_bread bread137)
     (at_kitchen_bread bread319)
     (at_kitchen_bread bread382)
     (at_kitchen_content content290)
     (at_kitchen_content content325)
     (at_kitchen_content content375)
     (not_allergic_gluten child474)
     (not_allergic_gluten child90)
     (not_allergic_gluten child233)
     (waiting child474 table166)
     (waiting child90 table125)
     (waiting child233 table63)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child474)
     (served child90)
     (served child233)
    )
  )
)
