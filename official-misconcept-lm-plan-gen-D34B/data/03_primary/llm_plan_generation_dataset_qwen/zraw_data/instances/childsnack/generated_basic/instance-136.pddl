; child-snack task with 2 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 377202

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child11 child142 - child
    bread49 bread412 - bread-portion
    content37 content204 - content-portion
    tray186 tray228 tray345 tray196 - tray
    table483 table222 table389 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray186 kitchen)
     (at tray228 kitchen)
     (at tray345 kitchen)
     (at tray196 kitchen)
     (at_kitchen_bread bread49)
     (at_kitchen_bread bread412)
     (at_kitchen_content content37)
     (at_kitchen_content content204)
     (not_allergic_gluten child11)
     (not_allergic_gluten child142)
     (waiting child11 table222)
     (waiting child142 table483)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child11)
     (served child142)
    )
  )
)
