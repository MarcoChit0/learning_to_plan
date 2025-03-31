; child-snack task with 2 children and 0.54 gluten factor 
; constant factor of 1.7
; random seed: 686402

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child142 child130 - child
    bread108 bread58 - bread-portion
    content0 content205 - content-portion
    tray350 - tray
    table135 table474 table195 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray350 kitchen)
     (at_kitchen_bread bread108)
     (at_kitchen_bread bread58)
     (at_kitchen_content content0)
     (at_kitchen_content content205)
     (no_gluten_bread bread58)
     (no_gluten_content content205)
     (allergic_gluten child142)
     (not_allergic_gluten child130)
     (waiting child142 table135)
     (waiting child130 table474)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child142)
     (served child130)
    )
  )
)
