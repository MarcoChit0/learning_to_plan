; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 256531

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child57 child382 - child
    bread373 bread378 - bread-portion
    content375 content198 - content-portion
    tray49 tray136 - tray
    table69 table56 table191 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray49 kitchen)
     (at tray136 kitchen)
     (at_kitchen_bread bread373)
     (at_kitchen_bread bread378)
     (at_kitchen_content content375)
     (at_kitchen_content content198)
     (not_allergic_gluten child57)
     (not_allergic_gluten child382)
     (waiting child57 table69)
     (waiting child382 table191)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child57)
     (served child382)
    )
  )
)
