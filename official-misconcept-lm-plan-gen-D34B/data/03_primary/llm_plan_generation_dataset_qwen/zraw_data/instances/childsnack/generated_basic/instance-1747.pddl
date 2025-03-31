; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 292843

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child257 - child
    bread483 - bread-portion
    content428 - content-portion
    tray93 tray279 tray88 - tray
    table136 table146 table50 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray93 kitchen)
     (at tray279 kitchen)
     (at tray88 kitchen)
     (at_kitchen_bread bread483)
     (at_kitchen_content content428)
     (not_allergic_gluten child257)
     (waiting child257 table146)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child257)
    )
  )
)
