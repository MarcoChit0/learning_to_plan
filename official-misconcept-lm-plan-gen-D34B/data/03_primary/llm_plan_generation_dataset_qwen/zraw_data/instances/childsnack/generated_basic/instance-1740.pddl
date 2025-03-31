; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 363799

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child408 - child
    bread17 - bread-portion
    content411 - content-portion
    tray197 tray276 tray65 - tray
    table415 table141 table310 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray197 kitchen)
     (at tray276 kitchen)
     (at tray65 kitchen)
     (at_kitchen_bread bread17)
     (at_kitchen_content content411)
     (not_allergic_gluten child408)
     (waiting child408 table310)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child408)
    )
  )
)
