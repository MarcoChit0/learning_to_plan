; child-snack task with 1 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 758545

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child125 - child
    bread158 - bread-portion
    content240 - content-portion
    tray192 tray484 tray117 - tray
    table275 table298 table42 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray192 kitchen)
     (at tray484 kitchen)
     (at tray117 kitchen)
     (at_kitchen_bread bread158)
     (at_kitchen_content content240)
     (not_allergic_gluten child125)
     (waiting child125 table42)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child125)
    )
  )
)
