; child-snack task with 2 children and 0.45 gluten factor 
; constant factor of 1.7
; random seed: 629324

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child109 child41 - child
    bread178 bread467 - bread-portion
    content13 content93 - content-portion
    tray451 tray372 tray198 - tray
    table160 table51 table117 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray451 kitchen)
     (at tray372 kitchen)
     (at tray198 kitchen)
     (at_kitchen_bread bread178)
     (at_kitchen_bread bread467)
     (at_kitchen_content content13)
     (at_kitchen_content content93)
     (not_allergic_gluten child109)
     (not_allergic_gluten child41)
     (waiting child109 table160)
     (waiting child41 table160)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child109)
     (served child41)
    )
  )
)
