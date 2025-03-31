; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.9
; random seed: 57937

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child308 child194 - child
    bread315 bread365 - bread-portion
    content336 content377 - content-portion
    tray460 tray410 tray181 tray37 - tray
    table242 table272 table380 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray460 kitchen)
     (at tray410 kitchen)
     (at tray181 kitchen)
     (at tray37 kitchen)
     (at_kitchen_bread bread315)
     (at_kitchen_bread bread365)
     (at_kitchen_content content336)
     (at_kitchen_content content377)
     (not_allergic_gluten child308)
     (not_allergic_gluten child194)
     (waiting child308 table272)
     (waiting child194 table272)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child308)
     (served child194)
    )
  )
)
