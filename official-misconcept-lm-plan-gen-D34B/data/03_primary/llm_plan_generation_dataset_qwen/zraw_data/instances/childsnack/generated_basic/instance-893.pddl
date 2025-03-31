; child-snack task with 3 children and 0.31 gluten factor 
; constant factor of 1.5
; random seed: 468667

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child368 child482 child222 - child
    bread399 bread68 bread379 - bread-portion
    content451 content190 content92 - content-portion
    tray202 - tray
    table451 table15 table96 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray202 kitchen)
     (at_kitchen_bread bread399)
     (at_kitchen_bread bread68)
     (at_kitchen_bread bread379)
     (at_kitchen_content content451)
     (at_kitchen_content content190)
     (at_kitchen_content content92)
     (not_allergic_gluten child368)
     (not_allergic_gluten child482)
     (not_allergic_gluten child222)
     (waiting child368 table15)
     (waiting child482 table96)
     (waiting child222 table451)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child368)
     (served child482)
     (served child222)
    )
  )
)
