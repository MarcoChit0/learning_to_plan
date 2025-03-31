; child-snack task with 1 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 218121

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child156 - child
    bread196 - bread-portion
    content75 - content-portion
    tray253 tray184 - tray
    table303 table389 table361 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray253 kitchen)
     (at tray184 kitchen)
     (at_kitchen_bread bread196)
     (at_kitchen_content content75)
     (not_allergic_gluten child156)
     (waiting child156 table361)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child156)
    )
  )
)
