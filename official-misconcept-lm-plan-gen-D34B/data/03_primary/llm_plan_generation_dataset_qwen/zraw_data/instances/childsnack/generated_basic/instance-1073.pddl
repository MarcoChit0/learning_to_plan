; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.3
; random seed: 438100

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child106 child95 - child
    bread241 bread281 - bread-portion
    content197 content446 - content-portion
    tray404 tray263 tray394 - tray
    table154 table49 table464 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray404 kitchen)
     (at tray263 kitchen)
     (at tray394 kitchen)
     (at_kitchen_bread bread241)
     (at_kitchen_bread bread281)
     (at_kitchen_content content197)
     (at_kitchen_content content446)
     (not_allergic_gluten child106)
     (not_allergic_gluten child95)
     (waiting child106 table464)
     (waiting child95 table154)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child106)
     (served child95)
    )
  )
)
