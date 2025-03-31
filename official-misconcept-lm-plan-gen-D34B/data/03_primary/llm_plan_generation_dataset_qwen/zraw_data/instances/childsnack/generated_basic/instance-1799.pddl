; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.5
; random seed: 109246

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child277 - child
    bread361 - bread-portion
    content184 - content-portion
    tray181 - tray
    table449 table465 table95 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray181 kitchen)
     (at_kitchen_bread bread361)
     (at_kitchen_content content184)
     (not_allergic_gluten child277)
     (waiting child277 table95)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child277)
    )
  )
)
