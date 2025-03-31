; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 837928

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child111 - child
    bread489 - bread-portion
    content325 - content-portion
    tray269 tray375 tray358 - tray
    table377 table177 table407 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray269 kitchen)
     (at tray375 kitchen)
     (at tray358 kitchen)
     (at_kitchen_bread bread489)
     (at_kitchen_content content325)
     (not_allergic_gluten child111)
     (waiting child111 table377)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child111)
    )
  )
)
