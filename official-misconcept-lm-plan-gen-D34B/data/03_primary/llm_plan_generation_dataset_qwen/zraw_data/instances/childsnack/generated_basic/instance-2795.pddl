; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.7
; random seed: 287940

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child57 child79 - child
    bread340 bread147 - bread-portion
    content320 content290 - content-portion
    tray461 tray62 - tray
    table126 table342 table17 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray461 kitchen)
     (at tray62 kitchen)
     (at_kitchen_bread bread340)
     (at_kitchen_bread bread147)
     (at_kitchen_content content320)
     (at_kitchen_content content290)
     (not_allergic_gluten child79)
     (not_allergic_gluten child57)
     (waiting child57 table17)
     (waiting child79 table342)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child57)
     (served child79)
    )
  )
)
