; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.9
; random seed: 109302

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child51 child13 - child
    bread164 bread458 - bread-portion
    content463 content415 - content-portion
    tray225 - tray
    table24 table104 table47 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray225 kitchen)
     (at_kitchen_bread bread164)
     (at_kitchen_bread bread458)
     (at_kitchen_content content463)
     (at_kitchen_content content415)
     (no_gluten_bread bread164)
     (no_gluten_content content463)
     (allergic_gluten child51)
     (not_allergic_gluten child13)
     (waiting child51 table104)
     (waiting child13 table47)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child51)
     (served child13)
    )
  )
)
