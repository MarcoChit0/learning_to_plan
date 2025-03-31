; child-snack task with 3 children and 0.23 gluten factor 
; constant factor of 1.9
; random seed: 762947

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child48 child7 child1 - child
    bread348 bread208 bread480 - bread-portion
    content332 content147 content271 - content-portion
    tray232 tray323 tray467 - tray
    table106 table231 table89 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray232 kitchen)
     (at tray323 kitchen)
     (at tray467 kitchen)
     (at_kitchen_bread bread348)
     (at_kitchen_bread bread208)
     (at_kitchen_bread bread480)
     (at_kitchen_content content332)
     (at_kitchen_content content147)
     (at_kitchen_content content271)
     (not_allergic_gluten child1)
     (not_allergic_gluten child7)
     (not_allergic_gluten child48)
     (waiting child48 table106)
     (waiting child7 table106)
     (waiting child1 table106)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child48)
     (served child7)
     (served child1)
    )
  )
)
