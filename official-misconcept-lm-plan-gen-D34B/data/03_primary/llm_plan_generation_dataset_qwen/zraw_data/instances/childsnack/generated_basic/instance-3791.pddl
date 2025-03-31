; child-snack task with 1 children and 0.46 gluten factor 
; constant factor of 1.9
; random seed: 629096

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child63 - child
    bread373 - bread-portion
    content217 - content-portion
    tray314 - tray
    table270 table81 table25 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray314 kitchen)
     (at_kitchen_bread bread373)
     (at_kitchen_content content217)
     (not_allergic_gluten child63)
     (waiting child63 table81)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child63)
    )
  )
)
