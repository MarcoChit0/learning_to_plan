; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.7
; random seed: 64802

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child281 child197 - child
    bread44 bread79 - bread-portion
    content306 content496 - content-portion
    tray319 tray70 - tray
    table244 table22 table420 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray319 kitchen)
     (at tray70 kitchen)
     (at_kitchen_bread bread44)
     (at_kitchen_bread bread79)
     (at_kitchen_content content306)
     (at_kitchen_content content496)
     (no_gluten_bread bread79)
     (no_gluten_content content306)
     (allergic_gluten child197)
     (not_allergic_gluten child281)
     (waiting child281 table420)
     (waiting child197 table22)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child281)
     (served child197)
    )
  )
)
