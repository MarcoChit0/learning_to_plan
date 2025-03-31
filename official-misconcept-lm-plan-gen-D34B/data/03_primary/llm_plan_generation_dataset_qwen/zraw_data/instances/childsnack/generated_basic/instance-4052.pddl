; child-snack task with 3 children and 0.22 gluten factor 
; constant factor of 1.9
; random seed: 770675

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child302 child402 child299 - child
    bread167 bread222 bread232 - bread-portion
    content470 content127 content220 - content-portion
    tray337 - tray
    table100 table65 table14 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray337 kitchen)
     (at_kitchen_bread bread167)
     (at_kitchen_bread bread222)
     (at_kitchen_bread bread232)
     (at_kitchen_content content470)
     (at_kitchen_content content127)
     (at_kitchen_content content220)
     (not_allergic_gluten child302)
     (not_allergic_gluten child299)
     (not_allergic_gluten child402)
     (waiting child302 table100)
     (waiting child402 table100)
     (waiting child299 table14)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child302)
     (served child402)
     (served child299)
    )
  )
)
