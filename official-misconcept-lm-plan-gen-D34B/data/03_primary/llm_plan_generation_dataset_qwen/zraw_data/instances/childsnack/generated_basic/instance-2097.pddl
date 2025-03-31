; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.5
; random seed: 73549

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child108 - child
    bread0 - bread-portion
    content17 - content-portion
    tray69 tray421 tray447 tray286 - tray
    table345 table460 table75 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray69 kitchen)
     (at tray421 kitchen)
     (at tray447 kitchen)
     (at tray286 kitchen)
     (at_kitchen_bread bread0)
     (at_kitchen_content content17)
     (not_allergic_gluten child108)
     (waiting child108 table460)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child108)
    )
  )
)
