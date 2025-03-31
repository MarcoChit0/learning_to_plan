; child-snack task with 1 children and 0.47 gluten factor 
; constant factor of 1.7
; random seed: 485644

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child125 - child
    bread132 - bread-portion
    content18 - content-portion
    tray440 tray264 tray297 tray181 - tray
    table420 table499 table235 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray440 kitchen)
     (at tray264 kitchen)
     (at tray297 kitchen)
     (at tray181 kitchen)
     (at_kitchen_bread bread132)
     (at_kitchen_content content18)
     (not_allergic_gluten child125)
     (waiting child125 table499)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child125)
    )
  )
)
