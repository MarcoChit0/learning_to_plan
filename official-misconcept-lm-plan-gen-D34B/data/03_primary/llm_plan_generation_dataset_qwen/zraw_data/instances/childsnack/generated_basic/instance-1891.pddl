; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.7
; random seed: 45777

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child456 child63 - child
    bread476 bread126 - bread-portion
    content452 content187 - content-portion
    tray298 tray460 - tray
    table32 table106 table481 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray298 kitchen)
     (at tray460 kitchen)
     (at_kitchen_bread bread476)
     (at_kitchen_bread bread126)
     (at_kitchen_content content452)
     (at_kitchen_content content187)
     (no_gluten_bread bread126)
     (no_gluten_content content187)
     (allergic_gluten child456)
     (not_allergic_gluten child63)
     (waiting child456 table32)
     (waiting child63 table106)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child456)
     (served child63)
    )
  )
)
