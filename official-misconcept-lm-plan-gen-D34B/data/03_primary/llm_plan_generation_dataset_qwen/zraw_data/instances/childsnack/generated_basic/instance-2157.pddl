; child-snack task with 1 children and 0.73 gluten factor 
; constant factor of 1.5
; random seed: 546837

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child357 - child
    bread205 - bread-portion
    content140 - content-portion
    tray126 - tray
    table93 table262 table110 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray126 kitchen)
     (at_kitchen_bread bread205)
     (at_kitchen_content content140)
     (not_allergic_gluten child357)
     (waiting child357 table110)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child357)
    )
  )
)
