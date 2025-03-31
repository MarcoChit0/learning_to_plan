; child-snack task with 2 children and 0.28 gluten factor 
; constant factor of 1.7
; random seed: 614186

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child307 child175 - child
    bread401 bread71 - bread-portion
    content447 content201 - content-portion
    tray398 - tray
    table246 table46 table469 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray398 kitchen)
     (at_kitchen_bread bread401)
     (at_kitchen_bread bread71)
     (at_kitchen_content content447)
     (at_kitchen_content content201)
     (not_allergic_gluten child307)
     (not_allergic_gluten child175)
     (waiting child307 table469)
     (waiting child175 table469)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child307)
     (served child175)
    )
  )
)
