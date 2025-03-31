; child-snack task with 1 children and 0.63 gluten factor 
; constant factor of 1.5
; random seed: 271829

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child53 - child
    bread279 - bread-portion
    content61 - content-portion
    tray321 - tray
    table253 table220 table193 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray321 kitchen)
     (at_kitchen_bread bread279)
     (at_kitchen_content content61)
     (not_allergic_gluten child53)
     (waiting child53 table193)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child53)
    )
  )
)
