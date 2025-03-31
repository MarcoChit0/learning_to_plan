; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 232379

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child292 child70 - child
    bread343 bread13 - bread-portion
    content94 content286 - content-portion
    tray358 tray107 - tray
    table33 table479 table81 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray358 kitchen)
     (at tray107 kitchen)
     (at_kitchen_bread bread343)
     (at_kitchen_bread bread13)
     (at_kitchen_content content94)
     (at_kitchen_content content286)
     (no_gluten_bread bread343)
     (no_gluten_content content286)
     (allergic_gluten child292)
     (not_allergic_gluten child70)
     (waiting child292 table33)
     (waiting child70 table479)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child292)
     (served child70)
    )
  )
)
