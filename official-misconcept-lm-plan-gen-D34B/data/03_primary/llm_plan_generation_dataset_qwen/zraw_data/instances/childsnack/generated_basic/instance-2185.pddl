; child-snack task with 1 children and 0.73 gluten factor 
; constant factor of 1.5
; random seed: 746592

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child265 - child
    bread98 - bread-portion
    content19 - content-portion
    tray431 - tray
    table177 table97 table341 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray431 kitchen)
     (at_kitchen_bread bread98)
     (at_kitchen_content content19)
     (not_allergic_gluten child265)
     (waiting child265 table97)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child265)
    )
  )
)
