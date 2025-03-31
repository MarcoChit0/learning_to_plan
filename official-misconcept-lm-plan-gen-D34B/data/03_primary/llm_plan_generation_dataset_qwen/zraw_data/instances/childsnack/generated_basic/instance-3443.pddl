; child-snack task with 1 children and 0.77 gluten factor 
; constant factor of 1.9
; random seed: 258570

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child4 - child
    bread410 - bread-portion
    content495 - content-portion
    tray60 tray472 tray290 tray274 - tray
    table460 table247 table450 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray60 kitchen)
     (at tray472 kitchen)
     (at tray290 kitchen)
     (at tray274 kitchen)
     (at_kitchen_bread bread410)
     (at_kitchen_content content495)
     (not_allergic_gluten child4)
     (waiting child4 table247)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child4)
    )
  )
)
