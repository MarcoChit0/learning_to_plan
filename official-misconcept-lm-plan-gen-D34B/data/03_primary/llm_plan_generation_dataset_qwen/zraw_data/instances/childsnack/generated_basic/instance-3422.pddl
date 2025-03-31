; child-snack task with 1 children and 0.77 gluten factor 
; constant factor of 1.9
; random seed: 781439

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child489 - child
    bread143 - bread-portion
    content73 - content-portion
    tray142 tray152 tray127 tray272 - tray
    table99 table253 table228 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray142 kitchen)
     (at tray152 kitchen)
     (at tray127 kitchen)
     (at tray272 kitchen)
     (at_kitchen_bread bread143)
     (at_kitchen_content content73)
     (not_allergic_gluten child489)
     (waiting child489 table99)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child489)
    )
  )
)
