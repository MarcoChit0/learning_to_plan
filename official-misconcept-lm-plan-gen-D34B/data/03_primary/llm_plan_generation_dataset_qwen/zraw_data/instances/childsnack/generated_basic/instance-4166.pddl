; child-snack task with 2 children and 0.28 gluten factor 
; constant factor of 1.7
; random seed: 153860

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child149 child20 - child
    bread258 bread36 - bread-portion
    content40 content321 - content-portion
    tray14 - tray
    table53 table176 table470 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray14 kitchen)
     (at_kitchen_bread bread258)
     (at_kitchen_bread bread36)
     (at_kitchen_content content40)
     (at_kitchen_content content321)
     (not_allergic_gluten child149)
     (not_allergic_gluten child20)
     (waiting child149 table176)
     (waiting child20 table53)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child149)
     (served child20)
    )
  )
)
