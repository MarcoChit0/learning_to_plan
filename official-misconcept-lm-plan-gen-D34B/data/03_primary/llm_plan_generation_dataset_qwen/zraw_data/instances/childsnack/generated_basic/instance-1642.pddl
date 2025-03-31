; child-snack task with 1 children and 0.71 gluten factor 
; constant factor of 1.3
; random seed: 854268

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child174 - child
    bread420 - bread-portion
    content474 - content-portion
    tray147 tray498 tray327 tray166 - tray
    table124 table141 table230 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray147 kitchen)
     (at tray498 kitchen)
     (at tray327 kitchen)
     (at tray166 kitchen)
     (at_kitchen_bread bread420)
     (at_kitchen_content content474)
     (not_allergic_gluten child174)
     (waiting child174 table141)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child174)
    )
  )
)
