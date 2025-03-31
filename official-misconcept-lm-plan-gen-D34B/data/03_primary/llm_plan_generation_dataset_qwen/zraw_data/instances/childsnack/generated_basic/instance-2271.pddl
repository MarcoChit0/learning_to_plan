; child-snack task with 2 children and 0.54 gluten factor 
; constant factor of 1.7
; random seed: 221790

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child171 child73 - child
    bread434 bread236 - bread-portion
    content373 content85 - content-portion
    tray236 - tray
    table75 table79 table397 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray236 kitchen)
     (at_kitchen_bread bread434)
     (at_kitchen_bread bread236)
     (at_kitchen_content content373)
     (at_kitchen_content content85)
     (no_gluten_bread bread236)
     (no_gluten_content content373)
     (allergic_gluten child73)
     (not_allergic_gluten child171)
     (waiting child171 table75)
     (waiting child73 table75)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child171)
     (served child73)
    )
  )
)
