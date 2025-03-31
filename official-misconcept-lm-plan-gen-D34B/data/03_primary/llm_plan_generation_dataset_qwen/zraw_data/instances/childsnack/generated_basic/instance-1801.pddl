; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.5
; random seed: 557675

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child312 - child
    bread40 - bread-portion
    content331 - content-portion
    tray484 - tray
    table360 table363 table34 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray484 kitchen)
     (at_kitchen_bread bread40)
     (at_kitchen_content content331)
     (not_allergic_gluten child312)
     (waiting child312 table363)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child312)
    )
  )
)
