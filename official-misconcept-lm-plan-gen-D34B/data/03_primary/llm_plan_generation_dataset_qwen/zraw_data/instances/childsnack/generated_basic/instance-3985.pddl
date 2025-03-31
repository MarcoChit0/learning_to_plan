; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.9
; random seed: 461172

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child89 - child
    bread421 - bread-portion
    content250 - content-portion
    tray100 tray14 tray373 - tray
    table196 table312 table10 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray100 kitchen)
     (at tray14 kitchen)
     (at tray373 kitchen)
     (at_kitchen_bread bread421)
     (at_kitchen_content content250)
     (not_allergic_gluten child89)
     (waiting child89 table312)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child89)
    )
  )
)
