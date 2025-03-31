; child-snack task with 1 children and 0.48 gluten factor 
; constant factor of 1.7
; random seed: 553120

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child73 - child
    bread386 - bread-portion
    content389 - content-portion
    tray108 tray228 tray98 - tray
    table45 table206 table400 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray108 kitchen)
     (at tray228 kitchen)
     (at tray98 kitchen)
     (at_kitchen_bread bread386)
     (at_kitchen_content content389)
     (not_allergic_gluten child73)
     (waiting child73 table400)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child73)
    )
  )
)
