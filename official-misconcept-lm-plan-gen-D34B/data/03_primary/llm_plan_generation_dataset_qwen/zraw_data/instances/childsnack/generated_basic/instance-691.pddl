; child-snack task with 1 children and 0.43 gluten factor 
; constant factor of 1.9
; random seed: 160336

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child68 - child
    bread270 - bread-portion
    content176 - content-portion
    tray214 tray391 tray85 tray61 - tray
    table85 table375 table443 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray214 kitchen)
     (at tray391 kitchen)
     (at tray85 kitchen)
     (at tray61 kitchen)
     (at_kitchen_bread bread270)
     (at_kitchen_content content176)
     (not_allergic_gluten child68)
     (waiting child68 table443)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child68)
    )
  )
)
