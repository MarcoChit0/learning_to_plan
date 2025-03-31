; child-snack task with 1 children and 0.23 gluten factor 
; constant factor of 1.7
; random seed: 706150

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child313 - child
    bread144 - bread-portion
    content53 - content-portion
    tray483 tray174 - tray
    table99 table479 table419 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray483 kitchen)
     (at tray174 kitchen)
     (at_kitchen_bread bread144)
     (at_kitchen_content content53)
     (not_allergic_gluten child313)
     (waiting child313 table479)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child313)
    )
  )
)
