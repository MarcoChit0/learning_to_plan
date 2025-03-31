; child-snack task with 1 children and 0.63 gluten factor 
; constant factor of 1.5
; random seed: 354347

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child394 - child
    bread448 - bread-portion
    content404 - content-portion
    tray303 - tray
    table31 table486 table260 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray303 kitchen)
     (at_kitchen_bread bread448)
     (at_kitchen_content content404)
     (not_allergic_gluten child394)
     (waiting child394 table31)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child394)
    )
  )
)
