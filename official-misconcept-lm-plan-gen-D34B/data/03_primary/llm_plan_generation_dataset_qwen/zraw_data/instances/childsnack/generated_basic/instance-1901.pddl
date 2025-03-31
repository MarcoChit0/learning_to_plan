; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.7
; random seed: 510213

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child106 child56 - child
    bread193 bread22 - bread-portion
    content478 content426 - content-portion
    tray459 tray426 - tray
    table432 table221 table22 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray459 kitchen)
     (at tray426 kitchen)
     (at_kitchen_bread bread193)
     (at_kitchen_bread bread22)
     (at_kitchen_content content478)
     (at_kitchen_content content426)
     (no_gluten_bread bread22)
     (no_gluten_content content426)
     (allergic_gluten child106)
     (not_allergic_gluten child56)
     (waiting child106 table432)
     (waiting child56 table22)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child106)
     (served child56)
    )
  )
)
