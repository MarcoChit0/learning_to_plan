; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.9
; random seed: 796532

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child233 child296 - child
    bread63 bread342 - bread-portion
    content443 content492 - content-portion
    tray447 tray415 tray352 - tray
    table224 table130 table33 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray447 kitchen)
     (at tray415 kitchen)
     (at tray352 kitchen)
     (at_kitchen_bread bread63)
     (at_kitchen_bread bread342)
     (at_kitchen_content content443)
     (at_kitchen_content content492)
     (no_gluten_bread bread342)
     (no_gluten_content content443)
     (allergic_gluten child233)
     (not_allergic_gluten child296)
     (waiting child233 table130)
     (waiting child296 table130)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child233)
     (served child296)
    )
  )
)
