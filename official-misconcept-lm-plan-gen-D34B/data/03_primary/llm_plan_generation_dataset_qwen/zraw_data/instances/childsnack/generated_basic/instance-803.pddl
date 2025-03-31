; child-snack task with 1 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 495483

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child401 - child
    bread257 - bread-portion
    content24 - content-portion
    tray413 tray92 - tray
    table392 table238 table92 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray413 kitchen)
     (at tray92 kitchen)
     (at_kitchen_bread bread257)
     (at_kitchen_content content24)
     (not_allergic_gluten child401)
     (waiting child401 table238)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child401)
    )
  )
)
