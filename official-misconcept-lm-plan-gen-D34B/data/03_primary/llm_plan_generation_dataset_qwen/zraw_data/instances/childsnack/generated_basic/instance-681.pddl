; child-snack task with 1 children and 0.43 gluten factor 
; constant factor of 1.9
; random seed: 811465

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child417 - child
    bread196 - bread-portion
    content334 - content-portion
    tray214 tray227 tray162 tray332 - tray
    table234 table340 table466 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray214 kitchen)
     (at tray227 kitchen)
     (at tray162 kitchen)
     (at tray332 kitchen)
     (at_kitchen_bread bread196)
     (at_kitchen_content content334)
     (not_allergic_gluten child417)
     (waiting child417 table466)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child417)
    )
  )
)
