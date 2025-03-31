; child-snack task with 1 children and 0.69 gluten factor 
; constant factor of 1.3
; random seed: 541148

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child261 - child
    bread492 - bread-portion
    content335 - content-portion
    tray394 tray372 tray160 tray134 - tray
    table357 table54 table324 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray394 kitchen)
     (at tray372 kitchen)
     (at tray160 kitchen)
     (at tray134 kitchen)
     (at_kitchen_bread bread492)
     (at_kitchen_content content335)
     (not_allergic_gluten child261)
     (waiting child261 table357)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child261)
    )
  )
)
