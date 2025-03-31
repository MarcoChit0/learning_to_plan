; child-snack task with 2 children and 0.28 gluten factor 
; constant factor of 1.7
; random seed: 152121

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child393 child354 - child
    bread214 bread46 - bread-portion
    content352 content326 - content-portion
    tray357 - tray
    table332 table420 table176 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray357 kitchen)
     (at_kitchen_bread bread214)
     (at_kitchen_bread bread46)
     (at_kitchen_content content352)
     (at_kitchen_content content326)
     (not_allergic_gluten child393)
     (not_allergic_gluten child354)
     (waiting child393 table332)
     (waiting child354 table420)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child393)
     (served child354)
    )
  )
)
