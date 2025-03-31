; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 662080

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child209 child95 - child
    bread110 bread200 - bread-portion
    content363 content397 - content-portion
    tray60 tray285 - tray
    table454 table179 table363 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray60 kitchen)
     (at tray285 kitchen)
     (at_kitchen_bread bread110)
     (at_kitchen_bread bread200)
     (at_kitchen_content content363)
     (at_kitchen_content content397)
     (not_allergic_gluten child209)
     (not_allergic_gluten child95)
     (waiting child209 table363)
     (waiting child95 table363)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child209)
     (served child95)
    )
  )
)
