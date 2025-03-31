; child-snack task with 1 children and 0.35 gluten factor 
; constant factor of 1.3
; random seed: 747205

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child43 - child
    bread349 - bread-portion
    content309 - content-portion
    tray143 tray48 tray168 tray470 - tray
    table135 table290 table83 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray143 kitchen)
     (at tray48 kitchen)
     (at tray168 kitchen)
     (at tray470 kitchen)
     (at_kitchen_bread bread349)
     (at_kitchen_content content309)
     (not_allergic_gluten child43)
     (waiting child43 table83)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child43)
    )
  )
)
