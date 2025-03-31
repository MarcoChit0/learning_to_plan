; child-snack task with 2 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 254002

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child159 child396 - child
    bread300 bread442 - bread-portion
    content456 content257 - content-portion
    tray395 tray284 tray357 - tray
    table488 table180 table225 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray395 kitchen)
     (at tray284 kitchen)
     (at tray357 kitchen)
     (at_kitchen_bread bread300)
     (at_kitchen_bread bread442)
     (at_kitchen_content content456)
     (at_kitchen_content content257)
     (not_allergic_gluten child159)
     (not_allergic_gluten child396)
     (waiting child159 table488)
     (waiting child396 table180)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child159)
     (served child396)
    )
  )
)
