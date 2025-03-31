; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 684135

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child185 child196 - child
    bread42 bread161 - bread-portion
    content350 content231 - content-portion
    tray222 tray118 - tray
    table255 table328 table317 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray222 kitchen)
     (at tray118 kitchen)
     (at_kitchen_bread bread42)
     (at_kitchen_bread bread161)
     (at_kitchen_content content350)
     (at_kitchen_content content231)
     (not_allergic_gluten child185)
     (not_allergic_gluten child196)
     (waiting child185 table328)
     (waiting child196 table328)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child185)
     (served child196)
    )
  )
)
