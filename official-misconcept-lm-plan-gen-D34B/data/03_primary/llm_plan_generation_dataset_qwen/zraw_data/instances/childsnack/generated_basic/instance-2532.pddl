; child-snack task with 1 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 744351

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child8 - child
    bread165 - bread-portion
    content183 - content-portion
    tray282 tray359 tray136 - tray
    table174 table394 table381 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray282 kitchen)
     (at tray359 kitchen)
     (at tray136 kitchen)
     (at_kitchen_bread bread165)
     (at_kitchen_content content183)
     (not_allergic_gluten child8)
     (waiting child8 table381)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child8)
    )
  )
)
