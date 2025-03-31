; child-snack task with 2 children and 0.45 gluten factor 
; constant factor of 1.7
; random seed: 15637

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child133 child228 - child
    bread413 bread235 - bread-portion
    content10 content281 - content-portion
    tray78 tray318 tray293 - tray
    table408 table293 table38 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray78 kitchen)
     (at tray318 kitchen)
     (at tray293 kitchen)
     (at_kitchen_bread bread413)
     (at_kitchen_bread bread235)
     (at_kitchen_content content10)
     (at_kitchen_content content281)
     (not_allergic_gluten child228)
     (not_allergic_gluten child133)
     (waiting child133 table38)
     (waiting child228 table408)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child133)
     (served child228)
    )
  )
)
