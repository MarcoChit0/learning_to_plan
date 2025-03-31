; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.5
; random seed: 717256

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child411 - child
    bread116 - bread-portion
    content228 - content-portion
    tray390 - tray
    table467 table331 table190 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray390 kitchen)
     (at_kitchen_bread bread116)
     (at_kitchen_content content228)
     (not_allergic_gluten child411)
     (waiting child411 table467)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child411)
    )
  )
)
