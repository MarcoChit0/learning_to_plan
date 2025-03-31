; child-snack task with 1 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 485491

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child165 - child
    bread308 - bread-portion
    content123 - content-portion
    tray43 tray187 - tray
    table384 table451 table386 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray43 kitchen)
     (at tray187 kitchen)
     (at_kitchen_bread bread308)
     (at_kitchen_content content123)
     (not_allergic_gluten child165)
     (waiting child165 table451)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child165)
    )
  )
)
