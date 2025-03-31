; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.9
; random seed: 974760

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child108 - child
    bread364 - bread-portion
    content216 - content-portion
    tray492 tray23 tray209 - tray
    table242 table204 table6 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray492 kitchen)
     (at tray23 kitchen)
     (at tray209 kitchen)
     (at_kitchen_bread bread364)
     (at_kitchen_content content216)
     (not_allergic_gluten child108)
     (waiting child108 table242)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child108)
    )
  )
)
