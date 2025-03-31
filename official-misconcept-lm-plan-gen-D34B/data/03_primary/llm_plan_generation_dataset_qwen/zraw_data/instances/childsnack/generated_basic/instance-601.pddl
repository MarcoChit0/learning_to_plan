; child-snack task with 1 children and 0.63 gluten factor 
; constant factor of 1.5
; random seed: 514003

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child490 - child
    bread23 - bread-portion
    content56 - content-portion
    tray13 - tray
    table318 table224 table302 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray13 kitchen)
     (at_kitchen_bread bread23)
     (at_kitchen_content content56)
     (not_allergic_gluten child490)
     (waiting child490 table224)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child490)
    )
  )
)
