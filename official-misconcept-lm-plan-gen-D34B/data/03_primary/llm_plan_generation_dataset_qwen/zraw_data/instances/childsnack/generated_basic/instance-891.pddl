; child-snack task with 3 children and 0.31 gluten factor 
; constant factor of 1.5
; random seed: 783442

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child107 child81 child56 - child
    bread119 bread131 bread475 - bread-portion
    content235 content382 content135 - content-portion
    tray468 - tray
    table108 table272 table221 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray468 kitchen)
     (at_kitchen_bread bread119)
     (at_kitchen_bread bread131)
     (at_kitchen_bread bread475)
     (at_kitchen_content content235)
     (at_kitchen_content content382)
     (at_kitchen_content content135)
     (not_allergic_gluten child81)
     (not_allergic_gluten child107)
     (not_allergic_gluten child56)
     (waiting child107 table108)
     (waiting child81 table272)
     (waiting child56 table272)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child107)
     (served child81)
     (served child56)
    )
  )
)
