; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.7
; random seed: 156472

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child25 child482 - child
    bread377 bread115 - bread-portion
    content141 content147 - content-portion
    tray439 tray111 - tray
    table4 table229 table418 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray439 kitchen)
     (at tray111 kitchen)
     (at_kitchen_bread bread377)
     (at_kitchen_bread bread115)
     (at_kitchen_content content141)
     (at_kitchen_content content147)
     (no_gluten_bread bread377)
     (no_gluten_content content147)
     (allergic_gluten child25)
     (not_allergic_gluten child482)
     (waiting child25 table4)
     (waiting child482 table4)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child25)
     (served child482)
    )
  )
)
