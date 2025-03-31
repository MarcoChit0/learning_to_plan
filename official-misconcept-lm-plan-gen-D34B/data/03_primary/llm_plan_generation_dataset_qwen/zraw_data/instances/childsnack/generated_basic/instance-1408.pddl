; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.7
; random seed: 119617

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child94 child453 - child
    bread20 bread33 - bread-portion
    content300 content376 - content-portion
    tray110 - tray
    table447 table255 table244 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray110 kitchen)
     (at_kitchen_bread bread20)
     (at_kitchen_bread bread33)
     (at_kitchen_content content300)
     (at_kitchen_content content376)
     (no_gluten_bread bread33)
     (no_gluten_content content376)
     (allergic_gluten child94)
     (not_allergic_gluten child453)
     (waiting child94 table255)
     (waiting child453 table447)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child94)
     (served child453)
    )
  )
)
