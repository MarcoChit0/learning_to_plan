; child-snack task with 3 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 965920

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child174 child76 child58 - child
    bread174 bread208 bread420 - bread-portion
    content488 content462 content208 - content-portion
    tray123 - tray
    table113 table166 table9 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray123 kitchen)
     (at_kitchen_bread bread174)
     (at_kitchen_bread bread208)
     (at_kitchen_bread bread420)
     (at_kitchen_content content488)
     (at_kitchen_content content462)
     (at_kitchen_content content208)
     (not_allergic_gluten child174)
     (not_allergic_gluten child76)
     (not_allergic_gluten child58)
     (waiting child174 table166)
     (waiting child76 table9)
     (waiting child58 table113)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child174)
     (served child76)
     (served child58)
    )
  )
)
