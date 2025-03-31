; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.9
; random seed: 683750

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child164 child425 - child
    bread168 bread427 - bread-portion
    content24 content13 - content-portion
    tray175 tray193 tray50 tray488 - tray
    table242 table451 table334 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray175 kitchen)
     (at tray193 kitchen)
     (at tray50 kitchen)
     (at tray488 kitchen)
     (at_kitchen_bread bread168)
     (at_kitchen_bread bread427)
     (at_kitchen_content content24)
     (at_kitchen_content content13)
     (not_allergic_gluten child425)
     (not_allergic_gluten child164)
     (waiting child164 table451)
     (waiting child425 table334)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child164)
     (served child425)
    )
  )
)
