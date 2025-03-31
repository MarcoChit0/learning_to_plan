; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 387919

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child154 - child
    bread469 - bread-portion
    content424 - content-portion
    tray478 tray300 tray214 - tray
    table114 table237 table412 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray478 kitchen)
     (at tray300 kitchen)
     (at tray214 kitchen)
     (at_kitchen_bread bread469)
     (at_kitchen_content content424)
     (not_allergic_gluten child154)
     (waiting child154 table114)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child154)
    )
  )
)
