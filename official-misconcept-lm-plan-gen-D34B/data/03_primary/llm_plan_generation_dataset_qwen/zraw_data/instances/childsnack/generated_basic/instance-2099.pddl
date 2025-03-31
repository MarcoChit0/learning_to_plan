; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.5
; random seed: 133876

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child169 - child
    bread208 - bread-portion
    content421 - content-portion
    tray206 tray94 tray323 tray208 - tray
    table200 table173 table349 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray206 kitchen)
     (at tray94 kitchen)
     (at tray323 kitchen)
     (at tray208 kitchen)
     (at_kitchen_bread bread208)
     (at_kitchen_content content421)
     (not_allergic_gluten child169)
     (waiting child169 table200)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child169)
    )
  )
)
