; child-snack task with 1 children and 0.2 gluten factor 
; constant factor of 1.9
; random seed: 148298

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child346 - child
    bread15 - bread-portion
    content305 - content-portion
    tray184 tray160 tray498 tray86 - tray
    table348 table43 table135 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray184 kitchen)
     (at tray160 kitchen)
     (at tray498 kitchen)
     (at tray86 kitchen)
     (at_kitchen_bread bread15)
     (at_kitchen_content content305)
     (not_allergic_gluten child346)
     (waiting child346 table43)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child346)
    )
  )
)
