; child-snack task with 1 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 401945

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child77 - child
    bread178 - bread-portion
    content483 - content-portion
    tray163 tray221 tray306 - tray
    table444 table386 table459 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray163 kitchen)
     (at tray221 kitchen)
     (at tray306 kitchen)
     (at_kitchen_bread bread178)
     (at_kitchen_content content483)
     (not_allergic_gluten child77)
     (waiting child77 table386)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child77)
    )
  )
)
