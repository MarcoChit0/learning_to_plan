; child-snack task with 1 children and 0.73 gluten factor 
; constant factor of 1.5
; random seed: 797445

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child412 - child
    bread331 - bread-portion
    content97 - content-portion
    tray424 - tray
    table98 table131 table216 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray424 kitchen)
     (at_kitchen_bread bread331)
     (at_kitchen_content content97)
     (not_allergic_gluten child412)
     (waiting child412 table98)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child412)
    )
  )
)
