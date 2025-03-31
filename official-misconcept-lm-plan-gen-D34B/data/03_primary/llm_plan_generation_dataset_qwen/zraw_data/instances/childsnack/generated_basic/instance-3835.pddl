; child-snack task with 1 children and 0.47 gluten factor 
; constant factor of 1.7
; random seed: 710848

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child86 - child
    bread217 - bread-portion
    content64 - content-portion
    tray34 tray295 tray10 tray365 - tray
    table434 table158 table466 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray34 kitchen)
     (at tray295 kitchen)
     (at tray10 kitchen)
     (at tray365 kitchen)
     (at_kitchen_bread bread217)
     (at_kitchen_content content64)
     (not_allergic_gluten child86)
     (waiting child86 table434)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child86)
    )
  )
)
