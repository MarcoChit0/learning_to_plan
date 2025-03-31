; child-snack task with 1 children and 0.63 gluten factor 
; constant factor of 1.5
; random seed: 589321

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child156 - child
    bread202 - bread-portion
    content433 - content-portion
    tray101 - tray
    table340 table489 table135 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray101 kitchen)
     (at_kitchen_bread bread202)
     (at_kitchen_content content433)
     (not_allergic_gluten child156)
     (waiting child156 table489)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child156)
    )
  )
)
