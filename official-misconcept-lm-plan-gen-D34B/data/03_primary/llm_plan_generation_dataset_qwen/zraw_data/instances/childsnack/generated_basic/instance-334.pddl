; child-snack task with 1 children and 0.2 gluten factor 
; constant factor of 1.9
; random seed: 523406

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child88 - child
    bread21 - bread-portion
    content413 - content-portion
    tray146 tray45 tray464 tray254 - tray
    table0 table79 table446 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray146 kitchen)
     (at tray45 kitchen)
     (at tray464 kitchen)
     (at tray254 kitchen)
     (at_kitchen_bread bread21)
     (at_kitchen_content content413)
     (not_allergic_gluten child88)
     (waiting child88 table446)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child88)
    )
  )
)
