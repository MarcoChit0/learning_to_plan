; child-snack task with 1 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 334228

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child39 - child
    bread285 - bread-portion
    content228 - content-portion
    tray286 - tray
    table93 table374 table37 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray286 kitchen)
     (at_kitchen_bread bread285)
     (at_kitchen_content content228)
     (not_allergic_gluten child39)
     (waiting child39 table37)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child39)
    )
  )
)
