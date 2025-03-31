; child-snack task with 2 children and 0.31 gluten factor 
; constant factor of 1.7
; random seed: 70747

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child36 child193 - child
    bread234 bread378 - bread-portion
    content443 content397 - content-portion
    tray432 tray305 tray42 - tray
    table67 table209 table327 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray432 kitchen)
     (at tray305 kitchen)
     (at tray42 kitchen)
     (at_kitchen_bread bread234)
     (at_kitchen_bread bread378)
     (at_kitchen_content content443)
     (at_kitchen_content content397)
     (not_allergic_gluten child36)
     (not_allergic_gluten child193)
     (waiting child36 table327)
     (waiting child193 table327)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child36)
     (served child193)
    )
  )
)
