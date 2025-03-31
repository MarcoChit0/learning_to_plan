; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.9
; random seed: 163739

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child47 child29 - child
    bread333 bread476 - bread-portion
    content406 content409 - content-portion
    tray163 tray478 tray32 - tray
    table487 table233 table157 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray163 kitchen)
     (at tray478 kitchen)
     (at tray32 kitchen)
     (at_kitchen_bread bread333)
     (at_kitchen_bread bread476)
     (at_kitchen_content content406)
     (at_kitchen_content content409)
     (no_gluten_bread bread333)
     (no_gluten_content content409)
     (allergic_gluten child29)
     (not_allergic_gluten child47)
     (waiting child47 table233)
     (waiting child29 table157)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child47)
     (served child29)
    )
  )
)
