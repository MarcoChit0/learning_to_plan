; child-snack task with 1 children and 0.71 gluten factor 
; constant factor of 1.3
; random seed: 571234

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child171 - child
    bread411 - bread-portion
    content39 - content-portion
    tray442 tray116 tray151 tray388 - tray
    table12 table124 table121 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray442 kitchen)
     (at tray116 kitchen)
     (at tray151 kitchen)
     (at tray388 kitchen)
     (at_kitchen_bread bread411)
     (at_kitchen_content content39)
     (not_allergic_gluten child171)
     (waiting child171 table12)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child171)
    )
  )
)
