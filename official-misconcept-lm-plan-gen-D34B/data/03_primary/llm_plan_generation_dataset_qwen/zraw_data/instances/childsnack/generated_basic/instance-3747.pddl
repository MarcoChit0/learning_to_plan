; child-snack task with 1 children and 0.45 gluten factor 
; constant factor of 1.9
; random seed: 240022

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child201 - child
    bread420 - bread-portion
    content103 - content-portion
    tray19 tray296 tray124 - tray
    table302 table38 table388 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray19 kitchen)
     (at tray296 kitchen)
     (at tray124 kitchen)
     (at_kitchen_bread bread420)
     (at_kitchen_content content103)
     (not_allergic_gluten child201)
     (waiting child201 table388)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child201)
    )
  )
)
