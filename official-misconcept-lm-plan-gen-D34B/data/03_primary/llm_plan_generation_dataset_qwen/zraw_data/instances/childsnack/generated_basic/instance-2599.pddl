; child-snack task with 1 children and 0.48 gluten factor 
; constant factor of 1.7
; random seed: 546334

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child172 - child
    bread194 - bread-portion
    content361 - content-portion
    tray426 tray107 tray222 - tray
    table179 table214 table82 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray426 kitchen)
     (at tray107 kitchen)
     (at tray222 kitchen)
     (at_kitchen_bread bread194)
     (at_kitchen_content content361)
     (not_allergic_gluten child172)
     (waiting child172 table82)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child172)
    )
  )
)
