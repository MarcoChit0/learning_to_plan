; child-snack task with 2 children and 0.31 gluten factor 
; constant factor of 1.7
; random seed: 73577

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child345 child284 - child
    bread456 bread11 - bread-portion
    content204 content299 - content-portion
    tray35 tray194 tray117 - tray
    table485 table25 table358 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray35 kitchen)
     (at tray194 kitchen)
     (at tray117 kitchen)
     (at_kitchen_bread bread456)
     (at_kitchen_bread bread11)
     (at_kitchen_content content204)
     (at_kitchen_content content299)
     (not_allergic_gluten child284)
     (not_allergic_gluten child345)
     (waiting child345 table485)
     (waiting child284 table485)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child345)
     (served child284)
    )
  )
)
