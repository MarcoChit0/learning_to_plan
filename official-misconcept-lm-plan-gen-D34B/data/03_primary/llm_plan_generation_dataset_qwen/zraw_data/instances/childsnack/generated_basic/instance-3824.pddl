; child-snack task with 1 children and 0.47 gluten factor 
; constant factor of 1.7
; random seed: 403453

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child173 - child
    bread351 - bread-portion
    content307 - content-portion
    tray448 tray284 tray421 tray244 - tray
    table221 table229 table459 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray448 kitchen)
     (at tray284 kitchen)
     (at tray421 kitchen)
     (at tray244 kitchen)
     (at_kitchen_bread bread351)
     (at_kitchen_content content307)
     (not_allergic_gluten child173)
     (waiting child173 table459)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child173)
    )
  )
)
