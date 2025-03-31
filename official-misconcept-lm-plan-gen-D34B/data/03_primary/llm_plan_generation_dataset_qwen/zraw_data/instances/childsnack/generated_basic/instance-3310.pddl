; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.3
; random seed: 979141

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child237 child70 - child
    bread329 bread256 - bread-portion
    content54 content20 - content-portion
    tray316 tray467 - tray
    table119 table115 table298 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray316 kitchen)
     (at tray467 kitchen)
     (at_kitchen_bread bread329)
     (at_kitchen_bread bread256)
     (at_kitchen_content content54)
     (at_kitchen_content content20)
     (not_allergic_gluten child70)
     (not_allergic_gluten child237)
     (waiting child237 table298)
     (waiting child70 table119)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child237)
     (served child70)
    )
  )
)
