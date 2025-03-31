; child-snack task with 2 children and 0.39 gluten factor 
; constant factor of 1.9
; random seed: 644187

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child417 child359 - child
    bread63 bread412 - bread-portion
    content232 content16 - content-portion
    tray167 tray101 - tray
    table467 table230 table196 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray167 kitchen)
     (at tray101 kitchen)
     (at_kitchen_bread bread63)
     (at_kitchen_bread bread412)
     (at_kitchen_content content232)
     (at_kitchen_content content16)
     (not_allergic_gluten child417)
     (not_allergic_gluten child359)
     (waiting child417 table467)
     (waiting child359 table196)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child417)
     (served child359)
    )
  )
)
