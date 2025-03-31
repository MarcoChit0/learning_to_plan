; child-snack task with 1 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 110292

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child240 - child
    bread231 - bread-portion
    content377 - content-portion
    tray389 tray484 tray46 - tray
    table371 table63 table45 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray389 kitchen)
     (at tray484 kitchen)
     (at tray46 kitchen)
     (at_kitchen_bread bread231)
     (at_kitchen_content content377)
     (not_allergic_gluten child240)
     (waiting child240 table45)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child240)
    )
  )
)
