; child-snack task with 1 children and 0.23 gluten factor 
; constant factor of 1.7
; random seed: 790722

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child441 - child
    bread447 - bread-portion
    content308 - content-portion
    tray459 tray311 - tray
    table466 table484 table372 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray459 kitchen)
     (at tray311 kitchen)
     (at_kitchen_bread bread447)
     (at_kitchen_content content308)
     (not_allergic_gluten child441)
     (waiting child441 table466)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child441)
    )
  )
)
