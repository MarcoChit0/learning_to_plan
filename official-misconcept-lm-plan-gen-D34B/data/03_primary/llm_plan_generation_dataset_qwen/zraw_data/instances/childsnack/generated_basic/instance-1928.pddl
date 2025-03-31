; child-snack task with 1 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 940290

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child156 - child
    bread351 - bread-portion
    content411 - content-portion
    tray233 tray443 tray230 - tray
    table163 table85 table52 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray233 kitchen)
     (at tray443 kitchen)
     (at tray230 kitchen)
     (at_kitchen_bread bread351)
     (at_kitchen_content content411)
     (not_allergic_gluten child156)
     (waiting child156 table163)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child156)
    )
  )
)
