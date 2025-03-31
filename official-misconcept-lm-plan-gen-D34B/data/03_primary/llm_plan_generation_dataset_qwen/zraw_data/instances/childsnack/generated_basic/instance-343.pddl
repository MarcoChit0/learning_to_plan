; child-snack task with 1 children and 0.2 gluten factor 
; constant factor of 1.9
; random seed: 918118

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child355 - child
    bread145 - bread-portion
    content367 - content-portion
    tray152 tray309 tray400 tray306 - tray
    table288 table35 table403 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray152 kitchen)
     (at tray309 kitchen)
     (at tray400 kitchen)
     (at tray306 kitchen)
     (at_kitchen_bread bread145)
     (at_kitchen_content content367)
     (not_allergic_gluten child355)
     (waiting child355 table403)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child355)
    )
  )
)
