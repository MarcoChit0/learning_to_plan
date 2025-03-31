; child-snack task with 1 children and 0.2 gluten factor 
; constant factor of 1.9
; random seed: 946376

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child12 - child
    bread318 - bread-portion
    content293 - content-portion
    tray284 tray17 tray153 tray179 - tray
    table17 table158 table484 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray284 kitchen)
     (at tray17 kitchen)
     (at tray153 kitchen)
     (at tray179 kitchen)
     (at_kitchen_bread bread318)
     (at_kitchen_content content293)
     (not_allergic_gluten child12)
     (waiting child12 table158)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child12)
    )
  )
)
