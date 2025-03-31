; child-snack task with 2 children and 0.38 gluten factor 
; constant factor of 1.9
; random seed: 78922

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child133 child164 - child
    bread343 bread408 - bread-portion
    content156 content370 - content-portion
    tray319 - tray
    table53 table286 table435 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray319 kitchen)
     (at_kitchen_bread bread343)
     (at_kitchen_bread bread408)
     (at_kitchen_content content156)
     (at_kitchen_content content370)
     (not_allergic_gluten child164)
     (not_allergic_gluten child133)
     (waiting child133 table435)
     (waiting child164 table435)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child133)
     (served child164)
    )
  )
)
