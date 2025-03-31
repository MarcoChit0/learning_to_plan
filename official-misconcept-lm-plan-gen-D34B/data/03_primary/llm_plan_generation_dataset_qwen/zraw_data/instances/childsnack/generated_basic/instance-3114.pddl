; child-snack task with 2 children and 0.51 gluten factor 
; constant factor of 1.9
; random seed: 955756

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child445 child416 - child
    bread84 bread373 - bread-portion
    content461 content220 - content-portion
    tray281 tray415 tray448 - tray
    table457 table383 table205 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray281 kitchen)
     (at tray415 kitchen)
     (at tray448 kitchen)
     (at_kitchen_bread bread84)
     (at_kitchen_bread bread373)
     (at_kitchen_content content461)
     (at_kitchen_content content220)
     (no_gluten_bread bread373)
     (no_gluten_content content461)
     (allergic_gluten child445)
     (not_allergic_gluten child416)
     (waiting child445 table205)
     (waiting child416 table383)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child445)
     (served child416)
    )
  )
)
