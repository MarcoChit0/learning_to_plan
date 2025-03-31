; child-snack task with 2 children and 0.28 gluten factor 
; constant factor of 1.7
; random seed: 800859

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child95 child404 - child
    bread41 bread241 - bread-portion
    content415 content288 - content-portion
    tray254 - tray
    table387 table66 table157 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray254 kitchen)
     (at_kitchen_bread bread41)
     (at_kitchen_bread bread241)
     (at_kitchen_content content415)
     (at_kitchen_content content288)
     (not_allergic_gluten child95)
     (not_allergic_gluten child404)
     (waiting child95 table387)
     (waiting child404 table66)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child95)
     (served child404)
    )
  )
)
