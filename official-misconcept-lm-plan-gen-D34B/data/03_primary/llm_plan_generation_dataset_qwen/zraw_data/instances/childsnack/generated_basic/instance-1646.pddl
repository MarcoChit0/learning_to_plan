; child-snack task with 1 children and 0.71 gluten factor 
; constant factor of 1.3
; random seed: 480555

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child312 - child
    bread234 - bread-portion
    content5 - content-portion
    tray198 tray420 tray167 tray445 - tray
    table167 table100 table160 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray198 kitchen)
     (at tray420 kitchen)
     (at tray167 kitchen)
     (at tray445 kitchen)
     (at_kitchen_bread bread234)
     (at_kitchen_content content5)
     (not_allergic_gluten child312)
     (waiting child312 table167)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child312)
    )
  )
)
